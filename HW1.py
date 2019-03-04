import random
from random import uniform

class Portfolio (object):
	def __init__(self, cash=0, stock=[], mutual_fund=[]):
		self.cash = cash
		self.stock = stock
		self.mutual_fund = mutual_fund
		
	def __str__(self):
		return "Overall Portfolio"
		
	def addCash (self, cash):
		self.cash += cash
	
	def withdrawCash (self, cash):
		self.cash -= cash

	def buyStock (self, share, stock):
		self.buystock = stock.price * int(share)
		if stock.price * int(share) > self.cash:
			print ("Insufficient funds")
		else:
			self.cash -= stock.price * int(share)
			print ("New stock is bought")
		
	def sellStock (self, symbol, share):
		if 1 * random.uniform (0.5, 1.5) > self.cash:
			print ("Insufficient funds")
		else:
			self.cash += 1 * random.uniform (0.5, 1.5) * int(share)
		
	def buyMutualFund(self, share, mf):
		if share * 1 > self.cash:
			print ("Insufficient funds")
		else:
			print ("New mutual fund is bought")
		    self.cash -= share * 1
			
	def sellMutualFund (self, symbol, share):
		self.cash += share * random.uniform (0.9, 1.2)
	
class Stock (Portfolio):
	def __init__(self, price, symbol):
		self.price = price
		self.symbol = symbol
		
class MutualFund (Portfolio):
	def __init__(self, symbol):
		self.symbol = symbol
	
portfolio.Portfolio ()
portfolio.addCash(300.50)
s = Stock(20, "HFH")
portfolio.buyStock(5, s)
mf1 = MutualFund("BRT")
mf2 = MutualFund("GHT")
portfolio.buyMutualFund(10.3, mf1)
portoflio.buyMutualFund(2, mf2)
print(portfolio)
portfolio.sellMutualFund("BRT", 3)
portfolio.sellStock("HFH", 1)
portfolio.withdrawCash(50)
portfolio.history()

	
		
			
	
	
		
		  
	
	