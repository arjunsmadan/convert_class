# convert_class
Convertible Bond Class. 

Includes:
1) Class structure to create a convertible bond object
2) Bond + option pricing function
3) Binomial tree recursive pricing function
4) Black-Scholes derived greeks (with appropriate scaling to represent the convertible bond greeks)

Shown Uses:
1) Test cases of individual convertible bond objects
2) Current theoretical prices & greeks for CORZ 2031 0s up 42.5 and 2029 3s up 30, CRWV 2031 1.75% up 25, and AKAM 2027 0.375% up 30
3) Gamma scalping backtest and PNL attribution for AKAM 2027 0.375% up 30
