from math import exp, log, sqrt, floor, pi
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt

'''
Convertible Bond class
Inputs:
	initial_stock_price - represents stock price at issue, used to calculate conversion ratio
	current_stock_price - stock price at the time of valuation
	conversion_premium - conversion premium % (e.g. 40)
	coupon - annual coupon % (e.g. 3.5)
	maturity - maturity of the bond
	time_to_maturity - time in year / decimal format to maturity (e.g. 2 years 3 months = 2.25)
	risk_free_rate - risk free rate, quoted as a percent
	credit_spread - credit spread, in bps
	equity_vol - underlying equity annualized volatility, as a %
	div_yield - underlying equity dividend yield, as a %
	costofborrow - borrow cost, in bps

Assumptions in the class: 
	Semiannual coupon payments
	Non-call life
	No bankrupcy risk

	Bond + Option Black Scholes pricing model
		Bond pricing - PV of cashflows from the debt
		Option pricing - Black Scholes, assuming an effective dividend yield of div yield - borrow cost
		Assumes option component behaves like a European equity call

	Binomial Pricing Model
		Steps calculated for once a month unless specicied as input

'''

class ConvertibleBond:
	def __init__(self, initial_stock_price, current_stock_price, conversion_premium, coupon, maturity, time_to_maturity, risk_free_rate, credit_spread, equity_vol, div_yield, costofborrow):
		self.initial_stock_price = initial_stock_price
		self.current_stock_price = current_stock_price
		self.conversion_premium = conversion_premium / 100 #% input
		self.coupon = coupon / 100 #% input
		self.maturity = maturity
		self.time_to_maturity = time_to_maturity
		self.risk_free_rate = risk_free_rate / 100 #% input
		self.credit_spread = credit_spread / 10000 #bps input
		self.equity_vol = equity_vol / 100 #% input
		self.div_yield = div_yield / 100 #% input
		self.costofborrow = costofborrow / 10000 #bps input
		self.par = 1000 #Par value of the bond

		#Derived values
		self.conversion_price = self.initial_stock_price * (1 + self.conversion_premium)
		self.conversion_ratio = self.par / self.conversion_price

	def bond_floor(self):
		total_rate = self.risk_free_rate + self.credit_spread
		freq = 2 #Semiannual coupon payments
		full_periods = int(self.time_to_maturity * freq)
		partial = (self.time_to_maturity * freq) - full_periods
		pv_coupons = 0
		coupon_payment = self.par * self.coupon / freq

		#full coupons
		for t in range(1, full_periods + 1):
			discount_factor = 1 / ((1 + total_rate / freq) ** t)
			pv_coupons += coupon_payment * discount_factor

		#Frational coupons. Assume accruing coupon payments. Remove for textbook staggered coupon payments.
		#''''
		if partial > 0:
			t = full_periods + partial
			discount_factor = 1 / ((1 + total_rate / freq) ** t)
			pv_coupons += coupon_payment * partial * discount_factor
		
		pv_par = self.par / ((1 + total_rate / freq) ** (self.time_to_maturity * freq))
		#'''
		
		#pv_par = self.par / ((1 + total_rate / freq) ** full_periods)
	
		total_value = pv_coupons + pv_par

		return total_value

	def BS_option_value(self): 
		current_stock_price = self.current_stock_price
		strike = self.conversion_price
		time_to_maturity = self.time_to_maturity
		risk_free_rate = self.risk_free_rate
		equity_vol = self.equity_vol
		effective_div_yield = self.div_yield + self.costofborrow
		conversion_ratio = self.conversion_ratio

		#Black-Scholes d1 and d2
		d1 = (log(current_stock_price / strike) + (risk_free_rate - effective_div_yield + 0.5*equity_vol ** 2) * time_to_maturity) / (equity_vol * sqrt(time_to_maturity))
		d2 = d1 - equity_vol * sqrt(time_to_maturity)

		#CDF of d1 and d2, standard normal distribution
		N_d1 = norm.cdf(d1)
		N_d2 = norm.cdf(d2)

		#BS model adjusted for effective div yield
		calloption_value = current_stock_price * exp(effective_div_yield * time_to_maturity * -1) * N_d1 - strike * exp(risk_free_rate * time_to_maturity * -1) * N_d2

		#Scale value for 1 share by conversion ratio for 1 bond
		adj_calloption_value = calloption_value * self.conversion_ratio 
		return adj_calloption_value

	def BS_greeks(self):
		current_stock_price = self.current_stock_price
		strike = self.conversion_price
		time_to_maturity = self.time_to_maturity
		risk_free_rate = self.risk_free_rate
		equity_vol = self.equity_vol
		effective_div_yield = self.div_yield + self.costofborrow
		conversion_ratio = self.conversion_ratio

		#Black-Scholes d1 and d2
		d1 = (log(current_stock_price / strike) + (risk_free_rate - effective_div_yield + 0.5*equity_vol ** 2) * time_to_maturity) / (equity_vol * sqrt(time_to_maturity))
		d2 = d1 - equity_vol * sqrt(time_to_maturity)

		 #Option Greeks for 1 share
        delta = exp(effective_div_yield * time_to_maturity * -1) * norm.cdf(d1)
        gamma = exp(effective_div_yield * time_to_maturity * -1) / (current_stock_price * equity_vol * sqrt(time_to_maturity))* norm.pdf(d1)
        theta = (1 / 252) * (-(current_stock_price * equity_vol * exp(effective_div_yield * time_to_maturity * -1) / (2 * sqrt(time_to_maturity)) * norm.pdf(d1)) - risk_free_rate * strike * exp(risk_free_rate * time_to_maturity * -1) * norm.cdf(d2) + effective_div_yield * current_stock_price * exp(effective_div_yield * time_to_maturity * -1) * norm.cdf(d1) )
        vega = (1/100) * current_stock_price * exp(effective_div_yield * time_to_maturity * -1) * sqrt(time_to_maturity) * norm.pdf(d1)

		return { 
			 "delta": round(delta * conversion_ratio / 10 / current_stock_price, 4),
            "gamma": round(gamma * conversion_ratio / 10 / (current_stock_price ** 2), 4), 
            "theta": round(theta * conversion_ratio / 10, 4),
            "vega": round(vega * conversion_ratio / 10, 4)
		}

	def BS_total_value(self):
		return round((self.bond_floor() + self.BS_option_value()) / 10, 2)

	def binomial_convert_value(self, steps = None):
		current_stock_price = self.current_stock_price
		conversion_price = self.conversion_price
		conversion_ratio = self.conversion_ratio
		coupon = self.coupon
		time_to_maturity = self.time_to_maturity
		risk_free_rate = self.risk_free_rate
		credit_spread = self.credit_spread
		equity_vol = self.equity_vol
		div_yield = self.div_yield
		costofborrow = self.costofborrow
		par = self.par

		#Default 1 step per month unless user specifies
		if steps is None:
			steps = int(time_to_maturity * 12)

		dt = time_to_maturity / steps
		
		up = exp(equity_vol * sqrt(dt)) #scales annualized vol
		down = 1 / up

		#Risk neutral probability under effective div yield
		effective_div_yield = div_yield + costofborrow
		r = risk_free_rate + credit_spread
		
		p = (exp((r - effective_div_yield) * dt) - down) / (up - down)

		#Precompute underlying equity price tree
		equity_tree = [[0 for j in range(i + 1)] for i in range(steps + 1)]
		for i in range(steps + 1):
			for j in range(i + 1):
				equity_tree[i][j] = current_stock_price * (up ** j) * (down ** (i-j))

		#Initial terminal value of the convert
		convert_tree = [[0 for j in range(i + 1)] for i in range(steps + 1)]
		for j in range(steps + 1):
			stock = equity_tree[steps][j]
			converted_value = stock * conversion_ratio
			hold_value = par #Assumes no bankrupcy risk
			convert_tree[steps][j] = max(converted_value, hold_value)

		#Continuous coupon amount per step
        coupon_carry = par * coupon * dt

		#Backward induction to complete the binomial tree
		for i in range(steps - 1, -1, -1):
			for j in range(i + 1):
				EV = p * convert_tree[i+1][j+1] + (1-p) * convert_tree[i+1][j]
				EV = (EV + coupon_carry) * exp(-r * dt) #add coupon and discount
				stock = equity_tree[i][j]
				converted_value = stock * conversion_ratio

				value = max(EV, converted_value)
				convert_tree[i][j] = value

		return round(convert_tree[0][0] / 10, 2)

#cb = ConvertibleBond(initial_stock_price = 100, current_stock_price = 100, conversion_premium = 35, coupon = 3.5, maturity = 5, time_to_maturity = 5, risk_free_rate = 4.0, credit_spread = 200, costofborrow = 50, equity_vol = 30, div_yield = 2.0)
#print(cb.BS_total_value())
#print(cb.binomial_convert_value())
#print(cb.BS_greeks())

#Pricing Core Scientific 0s up 42.5 2031 notes
CORZ_issue = ConvertibleBond(initial_stock_price = 15.78, current_stock_price = 15.78, conversion_premium = 42.5, coupon = 0, maturity = 7, time_to_maturity = 7, risk_free_rate = 4.5, credit_spread = 200, costofborrow = 50, equity_vol = 55, div_yield = 0)
CORZ_now = ConvertibleBond(initial_stock_price = 15.78, current_stock_price = 18.89, conversion_premium = 42.5, coupon = 0, maturity = 7, time_to_maturity = 5.4, risk_free_rate = 4.5, credit_spread = 200, costofborrow = 50, equity_vol = 55, div_yield = 0)
x_bs = CORZ_issue.BS_total_value()
x_b = CORZ_issue.binomial_convert_value()
y_bs = CORZ_now.BS_total_value()
y_b = CORZ_now.binomial_convert_value()

print(f"CORZ 2031 0s up 42.5: At issue, BS: {x_bs}, binom: {x_b}")
print(f"CORZ 2031 0s up 42.5: Current, BS: {y_bs}, binom: {y_b}")


