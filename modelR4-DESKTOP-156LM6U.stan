data {
  int I;            // 観測値の数
  int T;            // 時点の数
  int S;            // 一周期の時点の数
  int C;            // 時点の数
  int TC;            // 時点の数
  vector[I] Y;         // 応答変数（観測値）ベクトル
  int<lower=1, upper=T> TI[I]; // 観測値の時点番号
  int<lower=1, upper=C> CI[I]; // 観測値の時点番号
  int<lower=1, upper=TC> TCI[I]; // 観測値の時点番号
  int<lower=1, upper=TC> TCTC[TC]; // 観測値の時点番号
  int<lower=1, upper=T> TTC[TC]; // 観測値の時点番号
  int<lower=1, upper=C> CTC[TC]; // 観測値の時点番号
  real<lower=0> S_SW;
}

parameters {
  vector<lower=0>[T] b;       //  現存量の推定値
  real le;
  vector[S-1] pe_base;       // 周期成分の推定値
  vector<lower=0>[TC] pc;       //  現存量の推定値
  real<lower=0> s_w;       // 水準成分の過程誤差の標準偏差
  real<lower=0> lambda;
  vector<lower=lambda>[C] lambdac;
}

transformed parameters{
  vector[T] pe;       // 周期成分の推定値
  vector[T] beta;     // 時点ごとの係数
  vector[T] p;   // 状態の推定値
  vector[T] alpha;  // 形状パラメーター
  vector[TC] alphac;  // 形状パラメーター
  for (s in 1:S-1) {
    pe[s] = pe_base[s];  // 周期成分の遷移
  }
  for (t in S:T) {
    pe[t] = -sum(pe[(t - 35):(t - 1)]);  // 周期成分の遷移
  }
  for (t in 1:T){
    beta[t] = le + pe[t];  // 水準＋周期性＋外因性
    p[t] = exp(beta[t]) * b[t];
  }
  for (t in 1:T){
    p[t] =  b[t];
  }
  for (t in 1:T){
    alpha[t] = lambda * p[t];
  }
  for (tc in 1:TC){
    alphac[tc] = lambdac[CTC[tc]] * pc[tc];
  }
}

model {
  for (t in 2:T) {
    b[t] ~ normal(b[t-1], s_w);  // 現存量の過程誤差の遷移
  }
  for(tc in 1:TC){  // 確率分布に従う観測値
    pc[tc] ~ gamma(alpha[TTC[tc]], lambda); // alphaとlambda
  }
  for(i in 1:I){  // 確率分布に従う観測値
    Y[i] ~ gamma(alphac[TCI[i]], lambdac[CI[i]]); // alphaとlambda
  }
  // 事前分布
  s_w ~ normal(0,S_SW); //0.0005not good
}

generated quantities {
  vector[I] log_lik;
   
  for (i in 1:I) {
    log_lik[i] = gamma_lpdf(Y[i] | alphac[TCI[i]], lambdac[CI[i]]);
  }
}

