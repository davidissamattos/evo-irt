// IRT 2PL with multiple tries
// Author: David Issa Mattos
// Date: 5 April 2021

data {
  int<lower=0> N; // size of the vector
  int<lower=0> y_succ[N]; // number of successful tries
  int<lower=0> N_tries[N]; // number of tries
  int p[N]; // test taker index(the model)
  int<lower=0> Np; // number of test takes (number of models)
  int item[N]; // item index of the test (the dataset)
  int<lower=0> Nitem; // number of items in the test
}


parameters {
 real b[Nitem]; // difficulty parameter
 real<lower=0> a[Nitem]; // discrimination parameter
 real theta[Np]; // ability of the test taker
}

model {
  real prob[N];

  //Weakly informative priors
  b ~ normal(0, 3);
  a ~ normal(0,3);
  theta ~ normal(0,3);

  //Linear gaussian model
  for(i in 1:N){
    real mu;
    mu = a[item[i]]*(theta[p[i]]- b[item[i]]);
    prob[i] = exp(mu)/(1+exp(mu));
  }
  y_succ ~ binomial(N_tries,prob);

}

generated quantities{
  vector[N] log_lik;
  vector[N] y_rep;
  for(i in 1:N){
    real mu;
    real prob;
    mu = a[item[i]]*(theta[p[i]]- b[item[i]]);
    prob = exp(mu)/(1+exp(mu));
    log_lik[i] = binomial_lpmf(y_succ[i] | N_tries[i], prob );
    y_rep[i] = binomial_rng(N_tries[i], prob);
  }
}
