#include <opencv2/opencv.hpp>
#include <pthread.h>
#include <arm_neon.h>
#include <papi.h>
#include <iostream>
#include <chrono>
#include <atomic>

using namespace cv;
using namespace std;

#define NUM_THREADS 4

Mat gray, output;

pthread_barrier_t start_barrier;
pthread_barrier_t end_barrier;

int start_rows[NUM_THREADS];
int end_rows[NUM_THREADS];

bool stop_flag = false;

// ---------- Worker Thread ----------
void* sobel_worker(void* arg) {

    int tid = *(int*)arg;

    while (true) {

        pthread_barrier_wait(&start_barrier);
        if (stop_flag) break;

        for (int y = start_rows[tid]; y < end_rows[tid]; y++) {

            uchar* r0 = gray.ptr<uchar>(y-1);
            uchar* r1 = gray.ptr<uchar>(y);
            uchar* r2 = gray.ptr<uchar>(y+1);
            uchar* outp = output.ptr<uchar>(y);

            for (int x = 1; x < gray.cols - 9; x += 8) {

                // sliding window loads (16 bytes)
                uint8x16_t r0_block = vld1q_u8(&r0[x-1]);
                uint8x16_t r1_block = vld1q_u8(&r1[x-1]);
                uint8x16_t r2_block = vld1q_u8(&r2[x-1]);

                uint8x8_t left0   = vget_low_u8(r0_block);
                uint8x8_t center0 = vext_u8(left0, vget_high_u8(r0_block), 1);
                uint8x8_t right0  = vext_u8(left0, vget_high_u8(r0_block), 2);

                uint8x8_t left1   = vget_low_u8(r1_block);
                uint8x8_t center1 = vext_u8(left1, vget_high_u8(r1_block), 1);
                uint8x8_t right1  = vext_u8(left1, vget_high_u8(r1_block), 2);

                uint8x8_t left2   = vget_low_u8(r2_block);
                uint8x8_t center2 = vext_u8(left2, vget_high_u8(r2_block), 1);
                uint8x8_t right2  = vext_u8(left2, vget_high_u8(r2_block), 2);

                int16x8_t L0 = vreinterpretq_s16_u16(vmovl_u8(left0));
                int16x8_t L1 = vreinterpretq_s16_u16(vmovl_u8(left1));
                int16x8_t L2 = vreinterpretq_s16_u16(vmovl_u8(left2));

                int16x8_t R0 = vreinterpretq_s16_u16(vmovl_u8(right0));
                int16x8_t R1 = vreinterpretq_s16_u16(vmovl_u8(right1));
                int16x8_t R2 = vreinterpretq_s16_u16(vmovl_u8(right2));

                int16x8_t C0 = vreinterpretq_s16_u16(vmovl_u8(center0));
                int16x8_t C2 = vreinterpretq_s16_u16(vmovl_u8(center2));

                int16x8_t gx =
                    vsubq_s16(
                        vaddq_s16(vaddq_s16(L0, L2), vshlq_n_s16(L1,1)),
                        vaddq_s16(vaddq_s16(R0, R2), vshlq_n_s16(R1,1))
                    );

                int16x8_t gy =
                    vsubq_s16(
                        vaddq_s16(vaddq_s16(L0, R0), vshlq_n_s16(C0,1)),
                        vaddq_s16(vaddq_s16(L2, R2), vshlq_n_s16(C2,1))
                    );

                int16x8_t mag = vaddq_s16(vabsq_s16(gx), vabsq_s16(gy));
                uint8x8_t res = vqmovun_s16(mag);

                vst1_u8(&outp[x], res);
            }
        }

        pthread_barrier_wait(&end_barrier);
    }

    return nullptr;
}

// ---------- MAIN ----------
int main(int argc, char** argv) {

    if (argc < 2) {
        cout << "Usage: ./sobel_opt <video> [frame_limit]\n";
        return -1;
    }

    int frame_limit = -1;
    if (argc >= 3) frame_limit = atoi(argv[2]);

    VideoCapture cap(argv[1]);
    if (!cap.isOpened()) {
        cout << "Video open failed\n";
        return -1;
    }

    Mat frame;
    cap.read(frame);
    cvtColor(frame, gray, COLOR_BGR2GRAY);

    output = Mat(gray.rows, gray.cols, CV_8UC1);

    int rows_per_thread = gray.rows / NUM_THREADS;
    for (int i=0;i<NUM_THREADS;i++) {
        start_rows[i] = max(1, i*rows_per_thread);
        end_rows[i] = (i==NUM_THREADS-1) ?
            gray.rows-1 : (i+1)*rows_per_thread;
    }

    pthread_barrier_init(&start_barrier,NULL,NUM_THREADS+1);
    pthread_barrier_init(&end_barrier,NULL,NUM_THREADS+1);

    pthread_t threads[NUM_THREADS];
    int tids[NUM_THREADS];

    for (int i=0;i<NUM_THREADS;i++) {
        tids[i]=i;
        pthread_create(&threads[i],NULL,sobel_worker,&tids[i]);
    }

    // ---------- PAPI ----------
    PAPI_library_init(PAPI_VER_CURRENT);
    int EventSet=PAPI_NULL;
    long long values[2];
    PAPI_create_eventset(&EventSet);
    PAPI_add_event(EventSet,PAPI_TOT_CYC);
    PAPI_add_event(EventSet,PAPI_L1_DCM);
    PAPI_start(EventSet);

    int frames=0;
    auto t0=chrono::high_resolution_clock::now();

    do {

        frames++;

        cvtColor(frame, gray, COLOR_BGR2GRAY);

        pthread_barrier_wait(&start_barrier);
        pthread_barrier_wait(&end_barrier);

        if (frames%50==0) {
            auto now=chrono::high_resolution_clock::now();
            double el=chrono::duration<double>(now-t0).count();
            cout<<"Progress: "<<frames<<" frames | FPS "<<frames/el<<endl;
        }

        if (frame_limit>0 && frames>=frame_limit) break;

    } while(cap.read(frame));

    auto t1=chrono::high_resolution_clock::now();
    double elapsed=chrono::duration<double>(t1-t0).count();

    stop_flag=true;
    pthread_barrier_wait(&start_barrier);

    for (int i=0;i<NUM_THREADS;i++)
        pthread_join(threads[i],NULL);

    PAPI_stop(EventSet,values);

    cout<<"\n========== RESULTS ==========\n";
    cout<<"Frames processed: "<<frames<<endl;
    cout<<"Elapsed time (s): "<<elapsed<<endl;
    cout<<"Average FPS: "<<frames/elapsed<<endl;
    cout<<"Total cycles: "<<values[0]<<endl;
    cout<<"Total L1 D-cache misses: "<<values[1]<<endl;
    cout<<"Average cycles per frame: "<<values[0]/frames<<endl;
    cout<<"Average L1 D-cache misses per frame: "<<values[1]/frames<<endl;
    cout<<"=============================\n";
}