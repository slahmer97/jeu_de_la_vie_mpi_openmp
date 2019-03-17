// Jeu de la vie avec sauvegarde de quelques itérations
// compiler avec gcc -O3 -march=native (et -fopenmp si OpenMP souhaité)

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <openmpi-x86_64/mpi.h>
#include <string.h>
#include <unistd.h>
#include <assert.h>

// hauteur et largeur de la matrice
#define HM 1200
#define LM 800

// nombre total d'itérations
#define ITER 10001
// multiple d'itérations à sauvegarder
#define SAUV 1000

#define DIFFTEMPS(a,b) \
(((b).tv_sec - (a).tv_sec) + ((b).tv_usec - (a).tv_usec)/1000000.)

/* tableau de cellules */
typedef char Tab[HM][LM];

//===============================ADDED====================================
#define LOCAL_ROWS 200

typedef struct{
    char local[LOCAL_ROWS][LM];
}my_struct;
#define MY_STRUCT_SIZE sizeof(char)*(LM*LOCAL_ROWS)
void init(Tab);
void calcnouv(Tab, Tab);

Tab t2,t1;
Tab tsauvegarde[1+ITER/SAUV];
int rank;
my_struct data;
int next_save = 0;
void init_data_distribution(int rank,int size);
const char* get_rank_start_ptr(Tab,int,int);
const char* get_next_rank_start_ptr(Tab,int,int);
const char* get_previous_rank_start_ptr(Tab,int,int);
void struct_init_cpy(my_struct* dest,const char*next,const char*previous,const char*local);
void save_my_struct(const char*,my_struct*);
void print_all_matix(const char*filename,Tab a);
void forward_data_to_master();
void next_evolution();
void communicate_data_with_neighbors();
int main(int argc,char**argv)
{
    MPI_Init(&argc,&argv);
    int size;
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    if(rank == 0)
        printf("\n size = %d \n",size);

    int next_neighbor = 0;
    int previous_neighbor = 0;

    init(t1);
    print_all_matix("my_test/all",t1);
    init_data_distribution(rank,size);
    MPI_Finalize();
    return( 0 );
}
void init_data_distribution(int rank,int size){
    MPI_Datatype mpi_my_struct_type;
    MPI_Type_contiguous(MY_STRUCT_SIZE,MPI_CHAR,&mpi_my_struct_type);
    MPI_Type_commit(&mpi_my_struct_type);

    if(rank == 0){
        const char* next = get_next_rank_start_ptr(t1,rank,size);
        const char* previous = get_previous_rank_start_ptr(t1,rank,size);
        const char* local = get_rank_start_ptr(t1,rank,size);
        struct_init_cpy(&data, next, previous,local);
        save_my_struct("my_test/rank0",&data);
        my_struct tmp;
        for (int machine = 1; machine < size; ++machine) {
            //parallel
            next = get_next_rank_start_ptr(t1,machine,size);
            previous = get_previous_rank_start_ptr(t1,machine,size);
            local = get_rank_start_ptr(t1,machine,size);
            struct_init_cpy(&tmp, next, previous,local);
            char file_name[30];
           // sprintf(file_name,"file%d",machine);
           // save_my_struct(file_name,&tmp);
            MPI_Send(&tmp,1,mpi_my_struct_type,machine,0,MPI_COMM_WORLD);
        }

    }
    else{
        char file_name[20];
        sprintf(file_name,"my_test/recv_%d",rank);
        MPI_Recv(&data,1,mpi_my_struct_type,0,0,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        save_my_struct(file_name,&data);
    }

}
void struct_init_cpy(my_struct* dest,const char next[],const char previous[], const char *local){


    for (int j = 0; j < LOCAL_ROWS ; ++j)
        for (int i = 0; i < LM; ++i)
            dest->local[j][i] = local[j*LM+i];
}
const char* get_rank_start_ptr(Tab t,int rank_,int _size){
    if(rank_ < _size)
     return &t[rank_*LOCAL_ROWS][0];
    else
        return (const char *)0;
}
const char* get_next_rank_start_ptr(Tab t,int rank_,int _size){
    const char* ret = 0;
    if(rank_ == _size -1) ret = &t[0][0];
    else if(rank_ >= 0 && rank_ <_size-1){
        ret =&t[(rank_*LOCAL_ROWS) + LOCAL_ROWS][0] ;
    }
    return ret;
}
const char* get_previous_rank_start_ptr(Tab t,int rank_,int _size){
    const char* ret = 0;
    if(rank_ == 0) ret = &t[HM-1][0];
    else if(rank_ > 0 && rank_ <= _size-1){
        ret =&t[(rank_*LOCAL_ROWS)-1][0] ;
    }
    return ret;
}

int nbvois(my_struct* t, int i, int j)
{
    int n=0;

    if( i>0 )
    {  /* i-1 */
        if( j>0 )
            if( t->local[i-1][j-1] )
                n++;
        if( t->local[i-1][j] )
            n++;
        if( j<LM-1 )
            if( t->local[i-1][j+1] )
                n++;
    }
    if( j>0 )
        if( t->local[i][j-1] )
            n++;
    if( j<LM-1 )
        if( t->local[i][j+1] )
            n++;
    if( i<HM-1 )
    {  /* i+1 */
        if( j>0 )
            if( t->local[i+1][j-1] )
                n++;
        if( t->local[i+1][j] )
            n++;
        if( j<LM-1 )
            if( t->local[i+1][j+1] )
                n++;
    }
    return( n );
}
void test1(){
    /*
    assert(get_rank_start_ptr(t1,0,size)== &t1[0][0]);
    assert(get_rank_start_ptr(t1,1,size)== &t1[LOCAL_ROWS*1][0]);
    assert(get_rank_start_ptr(t1,2,size)== &t1[LOCAL_ROWS*2][0]);
    assert(get_rank_start_ptr(t1,3,size)== &t1[LOCAL_ROWS*3][0]);
    assert(get_rank_start_ptr(t1,4,size)== &t1[LOCAL_ROWS*4][0]);

     ========================


     assert(get_lower_rank_start_ptr(t1,0,size) == &t1[HM-1][0]);
    assert(get_lower_rank_start_ptr(t1,1,size) == &t1[LOCAL_ROWS-1][0]);
    assert(get_lower_rank_start_ptr(t1,2,size) == &t1[LOCAL_ROWS*2 -1][0]);
    assert(get_lower_rank_start_ptr(t1,3,size) == &t1[LOCAL_ROWS*3 -1][0]);
    assert(get_lower_rank_start_ptr(t1,4,size) == &t1[LOCAL_ROWS*4 -1][0]);


     =======================
      assert(get_upper_rank_start_ptr(t1,0,size) == &t1[LOCAL_ROWS][0]);
    assert(get_upper_rank_start_ptr(t1,1,size) == &t1[LOCAL_ROWS*2][0]);
    assert(get_upper_rank_start_ptr(t1,2,size) == &t1[LOCAL_ROWS*3][0]);
    assert(get_upper_rank_start_ptr(t1,5,size) == &t1[LOCAL_ROWS*6][0]);
    assert(get_upper_rank_start_ptr(t1,8,size) == &t1[LOCAL_ROWS*9][0]);
    assert(get_upper_rank_start_ptr(t1,9,size) == &t1[LOCAL_ROWS*10][0]);
    assert(get_upper_rank_start_ptr(t1,size-1,size) == &t1[0][0]);
     */
}
void init(Tab t){
    srand(time(0));
    for(int i=0 ; i<HM ; i++)
        for(int j=0 ; j<LM ; j++ )
        {
            // t[i][j] = rand()%2;
            // t[i][j] = ((i+j)%3==0)?1:0;
            // t[i][j] = (i==0||j==0||i==h-1||j==l-1)?0:1;
            t[i][j] = 0;
        }
    t[10][10] = 1;
    t[10][11] = 1;
    t[10][12] = 1;
    t[9][12] = 1;
    t[8][11] = 1;

    t[55][50] = 1;
    t[54][51] = 1;
    t[54][52] = 1;
    t[55][53] = 1;
    t[56][50] = 1;
    t[56][51] = 1;
    t[56][52] = 1;


    t[201][10] = 1;
    t[201][11] = 1;
    t[201][12] = 1;
    t[201][12] = 1;

    t[400][0] = 1;
    t[400][10] = 1;

    t[401][0] = 1;
    t[401][1] = 1;
    t[401][2] = 1;



    t[402][0] = 1;
    t[402][1] = 1;
    t[402][2] = 1;

    t[420][0] = 1;
    t[420][2] = 1;
    t[420][20] = 1;



    t[600][0] = 1;
    t[600][2] = 1;
    t[600][4] = 1;

    t[600][5] = 1;
    t[600][10] = 1;


}

void save_my_struct(const char* filename,my_struct* my_struct1){

   // printf("%s , %8x\n",filename,my_struct1);

    FILE *f = fopen(filename, "w");


    fprintf(f, "------------------ LOCAL %d ------------------\n", rank);
    for(int x=0 ; x<HM ; x++)
    {
        fprintf(f,"[%d]",x);
        for(int y=0 ; y<LM ; y++)
            fprintf(f,"%c",my_struct1->local[x][y]?'*':'0');
        fprintf(f,"%c",'\n');
    }

    fclose(f);
}
void print_all_matix(const char*filename,Tab a){
    FILE *f = fopen(filename, "w");
    fprintf(f, "------------------GLOBAL_Matrix------------------\n");

    for(int x=0 ; x<HM ; x++)
    {
        for(int y=0 ; y<LM ; y++)
            fprintf(f,"%c",a[x][y]?'*':'0');
        fprintf(f,"%c",'\n');
    }

    fclose(f);

}

void forward_data_to_master(){
    Tab *recvbuff = NULL;
    if(rank == 0){
        recvbuff = tsauvegarde;
    }

    MPI_Gather(data.local,LOCAL_ROWS*LM,MPI_CHAR,recvbuff,LOCAL_ROWS*LM,MPI_CHAR,0,MPI_COMM_WORLD);

    if(rank == 0)
        next_save++;
}

