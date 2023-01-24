typedef signed char int8_t;
typedef char bool;

#define REG(address) *(volatile unsigned int*)(address)
#define REGF(address) *(volatile float*)(address)
#define REGC(address) *(volatile char*)(address)
#define REGC(address) *(volatile char*)(address)
#define DMEM_BASE  (0x10000000)
#define GPIO_BASE  (0x80030000)

#define LIDAR_IMAGE_WIDTH 1152
#define LIDAR_IMAGE_HEIGHT 1152
#define LIDAR_IMAGE_DEPTH 24
#define N_BUFFERS 20
#define BUFFERS_AVAIL_ADDR_OFFSET 234881024 /* 8*28*1024*1024 */

#define true 1
#define false 0

#define FUNC_PREPROCESS 0
#define FUNC_REFINE 1

char* base_addr = (char*)0x10000000;
unsigned int BUFFERS[N_BUFFERS] = {
    0,
    8*1*1024*1024,
    8*2*1024*1024,
    8*3*1024*1024,
    8*4*1024*1024,
    8*5*1024*1024,
    8*6*1024*1024,
    8*7*1024*1024,
    8*8*1024*1024,
    8*9*1024*1024,
    8*10*1024*1024,
    8*11*1024*1024,
    8*12*1024*1024,
    8*13*1024*1024,
    8*14*1024*1024,
    8*15*1024*1024,
    8*16*1024*1024,
    8*17*1024*1024,
    8*18*1024*1024,
    8*19*1024*1024,
};
bool* BUFFERS_AVAIL = 0;
unsigned int LIDAR_IMAGE_BUFFER = 8*20*1024*1024;
unsigned int RECORD_BUFFER = 8*24*1024*1024;
unsigned int RISCV_ARGS_BUFFER = 8*27*1024*1024;

void* alloc(){
    for(int i=0; i<N_BUFFERS; ++i){
        if(BUFFERS_AVAIL[i]){
            BUFFERS_AVAIL[i] = false;
            return (void*)(base_addr + BUFFERS[i]);
        }
    }
    return 0;
}

void mfree(void* ptr){
    int idx = ((unsigned int)ptr-(unsigned int)base_addr) / (8*1024*1024);
    BUFFERS_AVAIL[idx] = true;
}

void preprocess(volatile float* lidar_points, volatile int* n_points, float z_offset, int input_quant_scale, volatile int* offsets, volatile int8_t* intensities){
    int n_valid_points = 0;
    float scale = (float)(1 << input_quant_scale);
    for(int i=0; i<*n_points; ++i){
        int x = (int)(lidar_points[0]*10.0f+0.5f) + 576;
        int y = (int)(lidar_points[1]*-10.0f+0.5f) + 576;
        /*int z = (int)((lidar_points[2]+z_offset)*5.0f+0.5f);*/
        /* z_offset = 3.7 */
        int z = (int)(lidar_points[2]*5.0f+19.0f);
        if(x>=0 && x<1152 && y>=0 && y<1152 && z>=0 && z<24){
            float intensity = lidar_points[3]*scale+0.5f;
            /*intensities[0] = (intensity>127.0f? 127 : (int8_t)intensity);*/
            intensities[0] = (int8_t)intensity;
            if(intensities[0]==0) intensities[0] = 1;

            offsets[0] = y*(LIDAR_IMAGE_WIDTH*LIDAR_IMAGE_DEPTH) + x*LIDAR_IMAGE_DEPTH + z;
            ++offsets;

            ++n_valid_points;
            ++intensities;
        }
        lidar_points += 5;
    }
    *n_points = n_valid_points;
}

void quaternion_to_matrix(float* qt, float matrix[3][3]){
    float qt0_2 = qt[0] * qt[0];
    float qt1_2 = qt[1] * qt[1];
    float qt2_2 = qt[2] * qt[2];
    float qt3_2 = qt[3] * qt[3];
    float qt12 = qt[1] * qt[2];
    float qt13 = qt[1] * qt[3];
    float qt23 = qt[2] * qt[3];
    float qt01 = qt[0] * qt[1];
    float qt02 = qt[0] * qt[2];
    float qt03 = qt[0] * qt[3];
    float s = 2.0f / (qt0_2 + qt1_2 + qt2_2 + qt3_2);
    matrix[0][0] = 1.0f - s * (qt2_2 + qt3_2);
    matrix[0][1] = s * (qt12 - qt03);
    matrix[0][2] = s * (qt13 + qt02);
    matrix[1][0] = s * (qt12 + qt03);
    matrix[1][1] = 1.0f - s * (qt1_2 + qt3_2);
    matrix[1][2] = s * (qt23 - qt01);
    matrix[2][0] = s * (qt13 - qt02);
    matrix[2][1] = s * (qt23 + qt01);
    matrix[2][2] = 1.0f - s * (qt1_2 + qt2_2);
}

void rotate(float inp[3], float outp[3], float mx[3][3]){
    outp[0] = mx[0][0]*inp[0] + mx[0][1]*inp[1] + mx[0][2]*inp[2];
    outp[1] = mx[1][0]*inp[0] + mx[1][1]*inp[1] + mx[1][2]*inp[2];
    //outp[2] = mx[2][0]*inp[0] + mx[2][1]*inp[1] + mx[2][2]*inp[2];
}

void sort_predictions(volatile float* preds, int n_preds){
    if(n_preds==0) return;
    float* sorted_preds = (float*)alloc();
    sorted_preds[0] = preds[0];
    sorted_preds[1] = preds[1];
    sorted_preds[2] = preds[2];
    sorted_preds[3] = preds[3];
    sorted_preds[4] = preds[4];
    for(int i=1; i<n_preds; ++i){
        for(int j=i-1; j>=0; --j){
            if(sorted_preds[j*5]<preds[i*5]){
                if(j==0){
                    for(int k=i; k>0; --k){
                        sorted_preds[k*5] = sorted_preds[(k-1)*5];
                        sorted_preds[k*5+1] = sorted_preds[(k-1)*5+1];
                        sorted_preds[k*5+2] = sorted_preds[(k-1)*5+2];
                        sorted_preds[k*5+3] = sorted_preds[(k-1)*5+3];
                        sorted_preds[k*5+4] = sorted_preds[(k-1)*5+4];
                    }
                    sorted_preds[0] = preds[i*5];
                    sorted_preds[1] = preds[i*5+1];
                    sorted_preds[2] = preds[i*5+2];
                    sorted_preds[3] = preds[i*5+3];
                    sorted_preds[4] = preds[i*5+4];
                    break;
                }
            }
            else{
                for(int k=i; k>j+1; --k){
                    sorted_preds[k*5] = sorted_preds[(k-1)*5];
                    sorted_preds[k*5+1] = sorted_preds[(k-1)*5+1];
                    sorted_preds[k*5+2] = sorted_preds[(k-1)*5+2];
                    sorted_preds[k*5+3] = sorted_preds[(k-1)*5+3];
                    sorted_preds[k*5+4] = sorted_preds[(k-1)*5+4];
                }
                sorted_preds[(j+1)*5] = preds[i*5];
                sorted_preds[(j+1)*5+1] = preds[i*5+1];
                sorted_preds[(j+1)*5+2] = preds[i*5+2];
                sorted_preds[(j+1)*5+3] = preds[i*5+3];
                sorted_preds[(j+1)*5+4] = preds[i*5+4];
                break;
            }
        }
    }
    for(int i=0; i<n_preds; ++i){
        preds[i*5] = sorted_preds[i*5];
        preds[i*5+1] = sorted_preds[i*5+1];
        preds[i*5+2] = sorted_preds[i*5+2];
        preds[i*5+3] = sorted_preds[i*5+3];
        preds[i*5+4] = sorted_preds[i*5+4];
    }
    mfree((void*)sorted_preds);
}

void refine_predictions(volatile float* preds, volatile int8_t* input_image, volatile float* centroids, volatile float* confidence, volatile int* n_preds, volatile float* ego_translation, volatile float* ego_rotation, float max_dist, float fuzzy_dist, float fuzzy_rate, float refine_dist, int n_cutoff, bool is_pedestrian){
    for(int i=0; i<*n_preds; ++i){
        int x = (int)(centroids[i*2]+0.5f) + 64;
        int y = (int)(centroids[i*2+1]+0.5f) + 64;
        bool has_points = false;
        int sx = x>7? (x-7) : 0;
        int sy = y>7? (y-7) : 0;
        int ex = x<LIDAR_IMAGE_WIDTH-7? (x+7) : (LIDAR_IMAGE_WIDTH-1);
        int ey = y<LIDAR_IMAGE_HEIGHT-7? (y+7) : (LIDAR_IMAGE_HEIGHT-1);
        for(int ry=sy; ry<ey; ++ry){
            int offset = ry * LIDAR_IMAGE_WIDTH;
            for(int rx=sx; rx<ex; ++rx){
                if(input_image[offset+rx]>0){
                    has_points = true;
                    break;
                }
            }
            if(has_points) break;
        }
        if(has_points==0){
            confidence[y*(LIDAR_IMAGE_WIDTH)+x] *= 0.1f;
        }
    }

    float mx[3][3] = {};
    quaternion_to_matrix((float*)ego_rotation, mx);
    for(int i=0; i<*n_preds; ++i){
        float xyz[3] = { centroids[i*2] / 10.0f - 51.2f, -centroids[i*2+1] / 10.0f + 51.2f, 1.5f};
        float rxyz[3] = {};
        rotate(xyz, rxyz, mx);
        rxyz[0] += ego_translation[0];
        rxyz[1] += ego_translation[1];
        centroids[i*2] = rxyz[0];
        centroids[i*2+1] = rxyz[1];
    }

    volatile float* refined_preds = (volatile float*)alloc();
    int n_refined_preds = 0;
    for(int i=0; i<*n_preds; ++i){
        float dx = centroids[i*2] - ego_translation[0];
        float dy = centroids[i*2+1] - ego_translation[1];
        float d2 = dx*dx + dy*dy;
        if(d2>max_dist*max_dist) continue;
        refined_preds[n_refined_preds*5+1] = 1e10f;
        refined_preds[n_refined_preds*5+3] = centroids[i*2];
        refined_preds[n_refined_preds*5+4] = centroids[i*2+1];
        if(d2>fuzzy_dist*fuzzy_dist){
            refined_preds[n_refined_preds*5] = confidence[i] * fuzzy_rate;
            refined_preds[n_refined_preds*5+2] = confidence[i] * fuzzy_rate;
        }
        else{
            refined_preds[n_refined_preds*5] = confidence[i];
            refined_preds[n_refined_preds*5+2] = confidence[i];
        }
        ++n_refined_preds;
    }
    sort_predictions(refined_preds, n_refined_preds);
    n_refined_preds = (n_refined_preds>n_cutoff? n_cutoff : n_refined_preds);
    *n_preds = n_refined_preds;
    float m = 1.0f / refine_dist;
    for(int i=0; i<*n_preds && i<50; ++i){
        sort_predictions(refined_preds, *n_preds-i);
        preds[i*3] = refined_preds[3];
        preds[i*3+1] = refined_preds[4];
        preds[i*3+2] = refined_preds[0];
        refined_preds += 5;
        for(int n=0; n<*n_preds-i-1; ++n){
            float dx = refined_preds[n*5+3] - preds[i*3];
            float dy = refined_preds[n*5+4] - preds[i*3+1];
            float d2 = dx*dx + dy*dy;
            refined_preds[n*5+1] = (refined_preds[n*5+1]>d2? d2 : refined_preds[n*5+1]);
            if (is_pedestrian){
                if(refined_preds[n*5+1]==0.0f){
                    refined_preds[n*5] = refined_preds[n*5+2];
                }
                else{
                    float r = m * (refined_preds[n*5+1]>refine_dist? refine_dist: refined_preds[n*5+1]);
                    refined_preds[n*5] = refined_preds[n*5+2] * r;
                }
            }
            else{
                if(refined_preds[n*5+1]<0.4f){
                    refined_preds[n*5] = refined_preds[n*5+2] * 0.01f;
                }
                else{
                    float r = m * (refined_preds[n*5+1]>refine_dist? refine_dist: refined_preds[n*5+1]);
                    refined_preds[n*5] = refined_preds[n*5+2] * r;
                }
            }
        }
    }
    *n_preds = (n_refined_preds>50? 50 : n_refined_preds); // MOVE
    mfree((void*)refined_preds);
}

int main(void)
{
    REG(GPIO_BASE + 4) = 0; // 出力に設定
	REG(GPIO_BASE) = 0;

    volatile char* args = (volatile char*)(RISCV_ARGS_BUFFER + DMEM_BASE);

    volatile unsigned int* func = (volatile unsigned int*)args;

    if(*func==FUNC_PREPROCESS){
        volatile unsigned int* lidar_points_addr = (volatile unsigned int*)(args + 64);
        volatile float* lidar_points = (volatile float*)(*lidar_points_addr + DMEM_BASE);
        volatile int* n_points = (volatile int*)(args + 72);
        volatile float* z_offset = (volatile float*)(args + 80);
        volatile int* input_quant_scale = (volatile int*)(args + 88);
        volatile unsigned int* offsets_addr = (volatile unsigned int*)(args + 96);
        volatile int* offsets = (volatile int*)(*offsets_addr + DMEM_BASE);
        volatile unsigned int* intensities_addr = (volatile unsigned int*)(args + 104);
        volatile int8_t* intensities = (volatile int8_t*)(*intensities_addr + DMEM_BASE);
        preprocess(lidar_points, n_points, *z_offset, *input_quant_scale, offsets, intensities);
    }
    else if(*func==FUNC_REFINE){
        int8_t* input_image = (int8_t*)(LIDAR_IMAGE_BUFFER + DMEM_BASE);
        volatile unsigned int* preds_addr = (volatile unsigned int*)(args + 64);
        volatile float* preds = (volatile float*)(*preds_addr + DMEM_BASE);
        volatile unsigned int* centroids_addr = (volatile unsigned int*)(args + 72);
        volatile float* centroids = (volatile float*)(*centroids_addr + DMEM_BASE);
        volatile unsigned int* confidence_addr = (volatile unsigned int*)(args + 80);
        volatile float* confidence = (volatile float*)(*confidence_addr + DMEM_BASE);
        volatile int* n_preds = (volatile int*)(args + 88);
        volatile float* ego_translation = (volatile float*)(args + 96);
        volatile float* ego_rotation = (float*)(args + 112);
        volatile float* max_dist = (volatile float*)(args + 128);
        volatile float* fuzzy_dist = (volatile float*)(args + 136);
        volatile float* fuzzy_rate = (volatile float*)(args + 144);
        volatile float* refine_dist = (volatile float*)(args + 152);
        volatile int* n_cutoff = (volatile int*)(args + 160);
        volatile bool* is_pedestrian = (volatile bool*)(args + 168);
        refine_predictions(
            preds, input_image, centroids, confidence, n_preds, ego_translation, ego_rotation, *max_dist, *fuzzy_dist, *fuzzy_rate, *refine_dist, *n_cutoff, *is_pedestrian
        );
    }

	REG(GPIO_BASE) = 0x01; // 終了通知
	while(1) {}

	return 0;
}
