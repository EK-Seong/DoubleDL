function psi = price_sensitivity(t)
psi = 2*((t-5)^4/600+exp(-4*(t-5)^2)+t/10-2);