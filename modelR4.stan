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
  real<lower=0> b1;
  vector<lower=0>[T] d;       //  現存量の推定値
  // vector<lower=0, upper=1>[T] d_r;       //  現存量の推定値
  real le;
  vector[S-1] pe_base;       // 周期成分の推定値
  vector<lower=0>[TC] pc;       //  現存量の推定値
  // real<lower=0> mu;       // 水準成分の過程誤差の標準偏差
  real<lower=0> s_w;       // 水準成分の過程誤差の標準偏差
  vector<lower=0>[T] lambda;
  // vector<lower=0>[TC] lambdac;
  vector<lower=0, upper=1>[TC] lambdac_raw;
}

transformed parameters{
  vector[T] b;       //  現存量の推定値
  // vector[T] d;       //  現存量の推定値
  vector[T] pe;       // 周期成分の推定値
  vector[T] beta;     // 時点ごとの係数
  vector[T] p;   // 状態の推定値
  vector[T] alpha;  // 形状パラメーター
  vector[TC] alphac;  // 形状パラメーター
  vector[TC] lambdac;
  for (tc in 1:TC) {
    lambdac[tc] = lambda[TTC[tc]] + (lambda[TTC[tc]]*10 - lambda[TTC[tc]]) * lambdac_raw[tc];
  }
  // vector<lower=0>[TC] dlambda;
  
  // for (tc in 1:TC) {
  //   dlambda[tc] = lambdac[tc] - lambda[TTC[tc]];  // 周期成分の遷移
  // }
  b[1] = b1;
  for (t in 2:T) {
    b[t] = b[t-1] + p[t-1] - d[t-1];  // 現存量の過程誤差の遷移
   // b[t] = b[t-1] + p[t-1] - d[t-1];  // 現存量の過程誤差の遷移
  }
  for (s in 1:S-1) {
    pe[s] = pe_base[s];  // 周期成分の遷移
  }
  for (t in S:T) {
    pe[t] = -sum(pe[(t - 35):(t - 1)]);  // 周期成分の遷移
  }
  for (t in 1:T){
    // d[t] = b[t] * d_r[t];
    beta[t] = le + pe[t];  // 水準＋周期性＋外因性
    p[t] = exp(beta[t]) * b[t];
  }
  for (t in 1:T){
    alpha[t] = lambda[t] * p[t];
  }
  for (tc in 1:TC){
    alphac[tc] = lambdac[tc] * pc[tc];
  }
}

model {
  for (t in 1:T) {
    d[t] ~ normal(0, s_w);  // 現存量の過程誤差の遷移
  }
  for(tc in 1:TC){  // 確率分布に従う観測値
    pc[tc] ~ gamma(alpha[TTC[tc]], lambda[TTC[tc]]); // alphaとlambda
  }
  for(i in 1:I){  // 確率分布に従う観測値
    Y[i] ~ gamma(alphac[TCI[i]], lambdac[TCI[i]]); // alphaとlambda
  }
  // target += sum(lambdac_raw); // log Jacobian
  // 事前分布
  // b1 ~ normal(10,1);
  s_w ~ normal(0,S_SW); //0.0005not good
}

generated quantities {
  vector[I] log_lik;
   
  for (i in 1:I) {
    log_lik[i] = gamma_lpdf(Y[i] | alphac[TCI[i]], lambdac[TCI[i]]);
  }
}

