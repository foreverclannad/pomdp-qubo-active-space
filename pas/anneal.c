/* Feasible fixed-count, same-reference-role swaps; no physical labels inferred. */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

static uint64_t random64(uint64_t *s) {
    *s ^= *s >> 12; *s ^= *s << 25; *s ^= *s >> 27;
    return *s * UINT64_C(2685821657736338717);
}
static int allowed(const int *mask, const int *excluded, int ne, int k) {
    for (int e=0; e<ne; ++e) {
        int same=1;
        for (int t=0; t<k; ++t) if (!mask[excluded[e*k+t]]) { same=0; break; }
        if (same) return 0;
    }
    return 1;
}
/* Retain distinct evaluated states so duplicate QUBO winners never buy more SA. */
static void retain(const int *mask, int n, int k, double value,
                   const int *excluded, int ne, int capacity,
                   int *output, double *scores, int *count) {
    int slot=*count;
    if (slot==capacity) {
        slot=0;
        for(int i=1;i<capacity;++i) if(scores[i]<scores[slot]) slot=i;
        if(value<=scores[slot]) return;
    }
    if(!allowed(mask,excluded,ne,k)) return;
    for(int i=0;i<*count;++i) {
        int same=1;
        for(int j=0;j<k;++j) if(!mask[output[i*k+j]]) { same=0; break; }
        if(same) return;
    }
    int j=0;
    for(int i=0;i<n;++i) if(mask[i]) output[slot*k+j++]=i;
    scores[slot]=value;
    if(*count<capacity) ++*count;
}

int pas_anneal(int n, int k, int steps, int restarts, uint64_t seed,
               const double *linear, const double *pair,
               const int *roles, int electrons,
               const int *excluded, int ne, double temperature,
               int capacity, int *output, double *scores, uint64_t *completed) {
    *completed=0;
    if (k<=0 || k>n || restarts<1 || steps<restarts || steps%restarts || capacity<1) return -1;
    int sizes[3]={0,0,0}, take[3]={k,0,0};
    for(int i=0;i<n;++i) {
        int group=electrons<0 ? 0 : roles[i];
        if(group<0 || group>2) return -3;
        ++sizes[group];
    }
    if(electrons>=0) {
        if(electrons<sizes[1] || (electrons-sizes[1])%2) return -3;
        take[1]=sizes[1]; take[2]=(electrons-take[1])/2; take[0]=k-take[1]-take[2];
    }
    int inside[3], outside[3], movable[3], nm=0, a=0, b=k;
    for(int group=0;group<3;++group) {
        if(take[group]<0 || take[group]>sizes[group]) return -3;
        inside[group]=a; outside[group]=b; a+=take[group]; b+=sizes[group]-take[group];
        if(take[group]>0 && take[group]<sizes[group]) movable[nm++]=group;
    }
    int *order=malloc(n*sizeof(int)), *mask=malloc(n*sizeof(int));
    double *field=malloc(n*sizeof(double));
    if (!order || !mask || !field) { free(order); free(mask); free(field); return -2; }
    uint64_t state=seed ? seed : UINT64_C(20261004);
    int found=0, each=steps/restarts;
    double decay=pow(1e-4, 1.0/each);
    for (int restart=0; restart<restarts; ++restart) {
        for (int i=0;i<n;++i) { order[i]=i; mask[i]=0; }
        for (int i=n-1;i>0;--i) {
            int j=random64(&state)%(i+1), tmp=order[i]; order[i]=order[j]; order[j]=tmp;
        }
        int remaining[3]={take[0],take[1],take[2]};
        for(int i=0;i<n;++i) {
            int orbital=order[i], group=electrons<0 ? 0 : roles[orbital];
            if(remaining[group]>0) { mask[orbital]=1; --remaining[group]; }
        }
        int next_in[3]={inside[0],inside[1],inside[2]}, next_out[3]={outside[0],outside[1],outside[2]};
        for(int i=0;i<n;++i) {
            int group=electrons<0 ? 0 : roles[i];
            order[mask[i] ? next_in[group]++ : next_out[group]++]=i;
        }
        double value=0;
        for (int i=0;i<n;++i) {
            field[i]=linear[i];
            for (int t=0;t<k;++t) field[i]+=2*pair[i*n+order[t]];
            if (mask[i]) value+=linear[i]+(field[i]-linear[i])/2;
        }
        retain(mask,n,k,value,excluded,ne,capacity,output,scores,&found);
        double temp=temperature;
        for (int step=0;step<each;++step) {
            if(!nm) { temp*=decay; ++*completed; continue; }
            int group=movable[random64(&state)%nm];
            int a=inside[group]+random64(&state)%take[group];
            int b=outside[group]+random64(&state)%(sizes[group]-take[group]);
            int out=order[a], in=order[b];
            double delta=field[in]-field[out]-2*pair[in*n+out];
            double u=(random64(&state)>>11)*0x1.0p-53;
            mask[in]=1; mask[out]=0;
            retain(mask,n,k,value+delta,excluded,ne,capacity,output,scores,&found);
            if (delta>=0 || u<exp(delta/temp)) {
                order[a]=in; order[b]=out; value+=delta;
                for (int i=0;i<n;++i) field[i]+=2*(pair[i*n+in]-pair[i*n+out]);
            } else { mask[in]=0; mask[out]=1; }
            temp*=decay;
            ++*completed;
        }
    }
    free(order); free(mask); free(field);
    return found;
}
