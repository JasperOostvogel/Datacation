from collections import deque
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class SimResults :

    def __init__ ( self ):
        self.oldTime = 0
        self.oldTimeFP = 0                        # Store time of current simulation
        self.oldTimeSH = 0                        # Store time of current simulation
        self.oldTimeGI = 0                        # Store time of current simulation
        self.oldTimeRC = 0                        # Store time of current simulation
        
        self.sumQL = deque()                    # Store the total queue length
        self.waitingtimes = deque()             # Deque for storage of all waiting times
        self.notboarders = deque()              # Deque for storage of all not boarders
        self.stoptrain = deque()
        
        self.WaitingTimeResults = deque()       # Deque for a stations individual mean waiting time
        self.waitingTimeResults2 = deque()      # Deque for a stations individual mean waiting time squared
        self.VarWaitingTimeResults = deque()    # Deque for a stations individual waiting time variance
        
        self.QueueLengthResults = deque()       # Deque for a stations individual mean queue length
        self.QueueLengthResults2 = deque()      # Deque for a stations individual mean queue length squared
        self.VarQueueLengthResults = deque()    # Deque for a stations individual queue length variance
        
        
        self.notboardersrate = deque()          # Deque for a stations individual boarding rate
        self.notboardersrate2 = deque()         # Deque for a stations individual boarding rate squared
        
        self.xcount = deque()
        self.queueratios = deque()
        
        self.notboard1 = deque()                # Deque for a stations individual number of visitors that had to wait once
        self.notboard2 = deque()                # Deque for a stations individual number of visitors that had to wait twice
        self.notboard3 = deque()                # Deque for a stations individual number of visitors that had to wait thrice
        self.notboard4 = deque()                # Deque for a stations individual number of visitors that had to wait four or more times
        self.notboard5 = deque()                # Deque for a stations individual number of visitors that had to wait four or more times
        self.notboard6 = deque()                # Deque for a stations individual number of visitors that had to wait four or more times        
        
        self.totalcustomers = deque()           #Deque for a stations total customer
    
        self.stoptraincount = deque()
    
        self.table = pd.DataFrame()             #Dataframe that collects all the data from previous described deques

        self.Result = deque()
        self.ResultMean = deque()
        
    """Function to calculate queue lenght"""    
    def registerqueuelength(self, time, queuelength, station):
        self.oldTime = time                
        if station == 0:
            self.sumQL.append([queuelength*(time - self.oldTimeFP), (queuelength**2)*(time - self.oldTimeFP), station])
            self.oldTimeFP = time + 2
        elif station == 1:
            self.sumQL.append([queuelength*(time - self.oldTimeSH), (queuelength**2)*(time - self.oldTimeSH), station])
            self.oldTimeSH = time + 2
        elif station == 2:
            self.sumQL.append([queuelength*(time - self.oldTimeGI), (queuelength**2)*(time - self.oldTimeGI), station])
            self.oldTimeGI = time + 2
        elif station == 3:
            self.sumQL.append([queuelength*(time - self.oldTimeRC), (queuelength**2)*(time - self.oldTimeRC), station])
            self.oldTimeRC = time + 2
        for i in range (1,4):
            if queuelength <= 25*i:
               self.xcount.append(i)
            elif queuelength >75:
                self.xcount.append(4)
        
    """Function to register all waiting times at a station"""
    def registerwaitingtime (self , time, station):
        self.waitingtimes.append([time,station])            # Collect all the waiting times corresponding with each station
    
    """Function to register all service times at a station"""
    def registernotboarders(self, notboarders, station):
        self.notboarders.append([notboarders, station])     # Collect all the vistitors not able to board corresponding with each station
        
    """Function to register all times train could not enter at a station"""
    def registerstoptrain(self, station):
        self.stoptrain.append(station)     
    
    """Function to register """
    def registerallvisitors(self, FrogPond, SkunkHollow, GatorIsland, RaccoonCorner):
        self.totalcustomers.append(FrogPond)                # Collect total amount of visitors to FrogPond station
        self.totalcustomers.append(SkunkHollow)             # Collect total amount of visitors to SkunkHollow station
        self.totalcustomers.append(GatorIsland)             # Collect total amount of visitors to GatorIsland station
        self.totalcustomers.append(RaccoonCorner)           # Collect total amount of visitors to RaccoonCorner station

    """Function to calculate mean, mean squared and variance of waiting times at each station"""
    def getwaitingTimeResults(self):
        for i in range(4):                                  # Loop for all stations
            w=0                                             # Cumulative waiting time
            w2=0                                            # Cumulative squared waiting time
            Nw=0                                            # Number of waiting times
            for j in self.waitingtimes:                     # Loop over all registered waiting times
                if i == j[1]:                               # If the registered station corresponds with the station i the data will be stored for that station
                  if j[0] < 0:
                    Nw += 1                                 # Count number of waiting times
                  else:
                    Nw += 1  
                    w += j[0]                               # Add waiting time to cumulative waiting time of station i
                    w2 += j[0]**2                           # Add squared waiting time to cumulative waiting time of station i

            meanW = w/Nw                                    # Calculate mean waiting time for station i
            VarW = (w2/Nw) - meanW**2                       # Calculate waiting time variance for station i
            self.WaitingTimeResults.append(meanW)           # Store mean waiting time of station i in deque
            self.waitingTimeResults2.append(meanW**2)       # Store squared mean waiting time of station i in deque
            self.VarWaitingTimeResults.append(VarW)         # Store waiting time variance of station i in deque
        wtr = pd.Series(self.WaitingTimeResults)            # Make series of mean waiting time deque
        wtr2 = pd.Series(self.waitingTimeResults2)          # Make series of squared mean waiting time deque
        vwtr = pd.Series(self.VarWaitingTimeResults)        # Make series of waiting time variance deque
        self.table['Waiting Time'] = wtr.values             # Add mean waiting time series to dataframe
        self.table['Waiting Time Squared'] = wtr2.values    # Add squared mean wating time series to dataframe
        self.table['Var Waiting Time'] = vwtr.values        # Add waiting time variance to dataframe

    """Function to calculate mean, mean squared and variance of the queue length at each station"""    
    def getqueuelengthresults(self):
        for i in range(4):                                  # Loop for all stations
            sumQL = 0                                       # Cumulative queue length
            sumQL2 = 0                                      # Cumulative squared queue length
            for j in self.sumQL:                            # Loop over all resitered queue lengths
                if i == j[2]:                               # If the registered station corresponds with the station i the data will be stored for that station
                    sumQL += j[0]                           # Add queue length to cumulative waiting time of station i
                    sumQL2 += j[1]                          # Add squared queue length to cumulative waiting time of station i
            meanQL = sumQL/self.oldTime                     # Calculate mean queue length for station i
            VarQL = sumQL2/self.oldTime - meanQL**2         # Calculate queue length variance for station i
            self.QueueLengthResults.append(meanQL)          # Store mean queue length of station i in deque
            self.QueueLengthResults2.append(meanQL**2)      # Store squared mean queue length of station i in deque
            self.VarQueueLengthResults.append(VarQL)        # Store queue length variance of station i in deque
        qlr = pd.Series(self.QueueLengthResults)            # Make series of mean queue length deque
        qlr2 = pd.Series(self.QueueLengthResults2)          # Make series of squared mean queue length deque
        vqlr = pd.Series(self.VarQueueLengthResults)        # Make series of queue length variance deque
        self.table['Queue length'] = qlr.values             # Add mean queue length series to dataframe
        self.table['Queue length Squared'] = qlr2.values    # Add squared mean queue length series to dataframe
        self.table['Var Queue length'] = vqlr.values        # Add queue length variance to dataframe

    """Function to calculate utilization rate, squared utilization rate and number of customers at each station"""
    def getnotboardrate(self):                  
        for i in range(4):                                  # Loop for all stations
            notboardertimes = []
            for j in self.notboarders:                      # Loop over all registered 
                if i == j[1]:                               # If the registered station corresponds with the station i the data will be stored for that station
                    notboardertimes.append(j[0])            # Add service time to cumulative waiting time of station i
            numberofwaits = {}
            for x in notboardertimes:
                numberofwaits[x] = notboardertimes.count(x)
            temp = []             
            notboard = []
            for key, value in numberofwaits.items():                    
                temp.append(value)            
            for y in range(1,6):
                notboard.append(temp.count(y))                   
                temp = [value for value in temp if value != y]
            notboard.append(len(temp))            
            Bfactor = len(numberofwaits)/self.totalcustomers[i]             # Calculate the utilization factor
            self.notboardersrate.append(Bfactor)         # Store utilization factor of station i in deque
            self.notboardersrate2.append(Bfactor**2)     # Store squared utilization factor of station i in deque   
            self.notboard1.append(notboard[0])
            self.notboard2.append(notboard[1])
            self.notboard3.append(notboard[2])
            self.notboard4.append(notboard[3])
            self.notboard5.append(notboard[4])
            self.notboard6.append(notboard[5])
            
            
        br = pd.Series(self.notboardersrate)                # Make series of utilization rate deque
        br2 = pd.Series(self.notboardersrate2)              # Make series of squared utilization rate deque        
        tc = pd.Series(self.totalcustomers)                 # Make series of total customers deque
        
        nb1 = pd.Series(self.notboard1)
        nb2 = pd.Series(self.notboard2)
        nb3 = pd.Series(self.notboard3)
        nb4 = pd.Series(self.notboard4)
        nb5 = pd.Series(self.notboard5)
        nb6 = pd.Series(self.notboard6)
        
        self.table['Boarder rate'] = br.values              # Add utilization rate series to dataframe
        self.table['Boarder rate Squared'] = br2.values     # Add squared utilization rate series to dataframe
        self.table['Number visitors'] = tc.values           # Add total customers series to dataframe
        self.table['One wait visitors'] = nb1.values
        self.table['Two wait visitors'] = nb2.values
        self.table['Three wait visitors'] = nb3.values
        self.table['Four wait visitors'] = nb4.values
        self.table['Five wait visitors'] = nb5.values
        self.table['Six wait visitors'] = nb6.values

    """Function to calculate utilization rate, squared utilization rate and number of customers at each station"""
    def getstoptraincount(self):
        for i in range(4):
            self.stoptraincount.append(self.stoptrain.count(i))
        stc = pd.Series(self.stoptraincount)
        self.table['Stop train count'] = stc.values

    
    def getxcount(self):
        for i in range (1,5):            
            self.queueratios.append(self.xcount.count(i)/len(self.xcount))
        qr = pd.Series(self.queueratios)
        self.table['Queue Ratios'] = qr.values


    """Function to calculate the confidence interval"""    
    def CalculateCI(self,meansquared, mean, runs):       
        value = meansquared - mean**2                       #Formula to calculate the variance
        ci = 1.96*np.sqrt(value/runs)                       #Formula to calculate the confidence interval
        return ci                                           #Return confidence interval

    def Barplotmeans(self, means, str):     
        objects = ['Frog Pond', 'Skunk Hollow', 'Gator Island', 'Raccoon Corner'] #station numbers
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, means, align='center', alpha=0.5) #plot waiting time
        plt.xticks(y_pos, objects)
        plt.xlabel('Stations')
        plt.title(str)
        plt.show()

    def Notboarddistribution(self, means, str):
        objects = list(range(1,7))
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, means, align='center', alpha=0.5) #plot waiting time
        plt.xticks(y_pos, objects)
        plt.xlabel(str)
        plt.title('Times a visitor could not board at a station')
        plt.show()                                            