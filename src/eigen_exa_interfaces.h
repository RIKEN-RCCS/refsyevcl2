#pragma once

/*
 * EigenExa との interface を記述する
 * 富岳環境で富士通コンパイラでビルドするとモジュールのシンボルが . (ドット)を含むため
 * そのまま呼び出すと構造体と認識されてしまうので、一旦モジュールなしの subroutine または function でラップする
 */

/** EigenExa の機能を初期化する */
extern void eigen_libs_eigen_init_();
#define eigen_init() eigen_libs_eigen_init_()

/** EigenExa の機能を終了する */
extern void eigen_libs_eigen_free_();
#define eigen_free() eigen_libs_eigen_free_()

/** EigenExaで規定されるプロセスグリッド情報に対応するScaLAPACK(BLACS)のコンテキストを返す */
extern int eigen_blacs_eigen_get_blacs_context_();
#define eigen_get_blacs_context() eigen_blacs_eigen_get_blacs_context_();

/** EigenExaの主たるドライバルーチンである */
extern void eigen_sx_(const int*,const int*,double*,int*,double*,double*,int*,int*,int*,char*);
#define eigen_sx(n,nvec,a,lda,w,z,ldz,m_forward,m_backward,mode) eigen_sx_(n,nvec,a,lda,w,z,ldz,m_forward,m_backward,mode)

/** EigenExaのドライバルーチンである */
extern void eigen_libs_eigen_s_(int*,int*,double*,int*,double*,double*,int*,int*,int*,char*);
#define eigen_s(n,nvec,a,lda,w,z,ldz,m_forward,m_backward,mode) eigen_libs_eigen_s_(n,nvec,a,lda,w,z,ldz,m_forward,m_backward,mode)

/** EigenExaのバージョン情報を返す */
extern void eigen_libs0_eigen_get_version_(int*,char*,char*);
#define eigen_get_version(version,date,vcode) eigen_libs0_eigen_get_version_(version,date,vcode)

/** EigenExaのバージョン情報を標準出力する */
extern void eigen_libs0_eigen_show_version_();
#define eigen_show_version() eigen_libs0_eigen_show_version_()

/** EigenExaで推奨する配列サイズを返す */
/*
extern void eigen_libs_eigen_get_matdims_(int*,int*,int*,int*,int*,char*);
#define eigen_get_matdims(n,nx,ny,m_forward,m_backward,mode) eigen_libs_eigen_get_matdims_(n,nx,ny,m_forward,m_backward,mode)
*/
extern void eigen_libs_eigen_get_matdims_(const int*,int*,int*);
#define eigen_get_matdims(n,nx,ny) eigen_libs_eigen_get_matdims_(n,nx,ny)

/** 本サブルーチンはEigenExaが呼び出されている間に内部で動的に確保されるメモリサイズを返す */
extern int eigen_libs0_eigen_memory_internal_(int*,int*,int*,int*,int*);
#define eigen_memory_internal(n,lda,ldz,m1_opt,m0_opt) eigen_libs0_eigen_memory_internal_(n,lda,ldz,m1_opt,m0_opt)

/** eigen_init()によって生成されたMPIコミュニケータを返す */
extern void eigen_libs0_eigen_get_comm_(int*,int*,int*);
#define eigen_get_comm(comm,x_comm,y_comm) eigen_libs0_eigen_get_comm_(comm,x_comm,y_comm)

/** eigen_init()によって生成されたコミュニケータに関するプロセス数情報を返す */
extern void eigen_libs0_eigen_get_procs_(int*,int*,int*);
#define eigen_get_procs(procs,x_procs,y_procs) eigen_libs0_eigen_get_procs_(procs,x_procs,y_procs)

/** eigen_init()によって生成されたコミュニケータに関するプロセスID情報を返す */
extern void eigen_libs0_eigen_get_id_(int*,int*,int*);
#define eigen_get_id(id,x_id,y_id) eigen_libs0_eigen_get_id_(id,x_id,y_id)

/** 指定されたグローバルループ開始値に対応するローカルなループ構造におけるループ開始値を返す */
extern int eigen_libs0_eigen_loop_start_(int*,int*,int*);
#define eigen_loop_start(istart,nnod,inod) eigen_libs0_eigen_loop_start_(istart,nnod,inod)

/** 指定されたグローバルループ終端値に対応するローカルなループ構造におけるループ終端値を返す */
extern int eigen_libs0_eigen_loop_end_(int*,int*,int*);
#define eigen_loop_end(iend,nnod,inod) eigen_libs0_eigen_loop_end_(iend,nnod,inod)

/** eigen_loop_startとeigen_loop_endを結合させたサブルーチンであり,開始値・終端値を同時に返す */
extern void eigen_libs0_eigen_loop_info_(int*,int*,int*,int*,int*,int*);
#define eigen_loop_info(istart,iend,lstart,lend,nnod,inod) eigen_libs0_eigen_loop_info_(istart,iend,lstart,lend,nnod,inod)

/** ローカルカウンタが示すローカルインデックス値(1以上の値)に対応するグローバルインデックスを返す */
extern int eigen_libs0_eigen_translate_l2g_(int*,int*,int*);
#define eigen_translate_l2g(ictr,nnod,inod) eigen_libs0_eigen_translate_l2g_(ictr,nnod,inod)

/** グローバルカウンタが示すグローバルインデックス値(1以上の値)に対応するローカルインデックスを返す */
extern int eigen_libs0_eigen_translate_g2l_(int*,int*,int*);
#define eigen_translate_g2l(ictr,nnod,inod) eigen_libs0_eigen_translate_g2l_(ictr,nnod,inod)

/** 指定されたグローバルインデックス値(1以上の値)に対応するオーナープロセスのIDを返す */
extern int eigen_libs0_eigen_owner_node_(int*,int*,int*);
#define eigen_owner_node(ictr,nnod,inod) eigen_libs0_eigen_owner_node_(ictr,nnod,inod)

/** 当該プロセスが指定されたグローバルインデックス値(1以上の値)のオーナーの場合に,対応するローカルインデックスを返す オーナーでない場合は−1 を返す */
extern int eigen_libs0_eigen_owner_index_(int*,int*,int*);
#define eigen_owner_index(ictr,nnod,inod) eigen_libs0_eigen_owner_index_(ictr,nnod,inod)

/** 2次元プロセスIDを,グリッドメジャーに応じて基盤コミュニケータ上のプロセスIDに変換する */
extern int eigen_libs0_eigen_convert_id_xy2w_(int*,int*);
#define eigen_convert_id_xy2w(xinod,yinod) eigen_libs0_eigen_convert_id_xy2w_(xinod,yinod)

/** 基盤コミュニケータ上のプロセスIDを,グリッドメジャーに応じて2次元プロセスIDに変換する */
extern void eigen_libs0_eigen_convert_id_w2xy_(int*,int*,int*);
#define eigen_convert_id_w2xy(inod,xinod,yinod) eigen_libs0_eigen_convert_id_w2xy_(inod,xinod,yinod)

/** eigen exa のエラー情報を取得する */
extern void eigen_libs0_eigen_get_errinfo_(int*);
#define eigen_get_errinfo(info) eigen_libs0_eigen_get_errinfo_(info)

typedef struct {
  double r;
  double i;
} complex;

/** エルミート固有値問題を計算するためのEigenExaのドライバルーチンである */
extern void eigen_h(int*,int*,complex*,int*,double*,complex*,int*,int*,int*,char*);
#define eigen_h(n,nvec,a,lda,w,z,ldz,m_forward,m_backward,mode) eigen_h_(n,nvec,a,lda,w,z,ldz,m_forward,m_backward,mode)
