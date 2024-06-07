import pandas as pd
import random
import numpy as np
import time
import matplotlib.pyplot as plt
import heapq    #sort events
from dist.Distribution import Distribution
from ClassSimResults3 import SimResults
from scipy import stats
from collections import deque
from pathlib import Path
import os

start_time = time.time()
#Make a dataframe of the customers arriving per hour per station
ArrivalDistributions = {0: [450, 300, 275, 285, 310, 320, 280, 260, 290, 315, 285, 415],       #Frog Pond
                        1: [400, 350, 250, 275, 290, 305, 300, 280, 310, 320, 360, 405],       #Skunk Hollow
                        2: [325, 340, 260, 210, 240, 280, 290, 275, 295, 330, 395, 430],       #Gator Island
                        3: [385, 320, 280, 265, 290, 315, 300, 320, 280, 310, 360, 395]        #Raccoon Corner
                        }

DestinationChances = {0: [0, 40, 75, 100],            #Frog Pond
                      1: [37, 37, 78, 100],           #Skunk Hollow
                      2: [42, 71, 71, 100],           #Gator Island
                      3: [41, 69, 100, 100]}          #Raccoon Corner


ArrivalDistributions = pd.DataFrame(ArrivalDistributions) #Make above dataframe in Pandas
ArrivalDistributions = 60 / ArrivalDistributions #Determine arrival time per customer per hour per station



NrTrains = 2
Cars = [13,12,4,4,4,4] #number of cars of each train
Warm_up_time = 27 #warum-up time, so the first train has travelled a full round before the start of the day
T = 720 + Warm_up_time #duration of one day (12 hours)
Runs = 500
Delaychance = 10
Timetoboard = 2
Delaytime = (1/6)
Switchtime = (1/2)          #time between a departure and an arrival, if the station is 


class Station:
    
    def __init__(self):        
        self.visitors = deque()
        self.status = 0
        self.deptime = 0
        
    def MakeStationQueue(self, j):  #make the deque for the whole day
        t = Warm_up_time
        while t < T:        
            for i in range(0,12): #12 hours per day with different Arrivaldistributions
                if t >= (60 * i) + Warm_up_time and t < (60 * (i+1)) + Warm_up_time: #Choose the correct distribution per hour per station
                    ArrivalDis = ArrivalDistributions[j][i] 
                    ArrivalTime = Distribution(stats.expon ( scale = ArrivalDis)).rvs() #schedule next arrival at station
                    Arrival_Station = t + ArrivalTime #save aarivaltime for a visitor of the themepark
            x = random.randint(0,100)   #decide the destination of the visitor
            if x <= DestinationChances[j][0]:
                Destination = 0
            elif x > DestinationChances[j][0] and x <= DestinationChances[j][1]:
                Destination = 1            
            elif x > DestinationChances[j][1] and x <= DestinationChances[j][2]:
                Destination = 2
            else:
                Destination = 3
            self.visitors.append([Arrival_Station, Destination])     #append to the deque for all the customers arrivaltimes and destination of one day
            t = Arrival_Station                 #update time
        self.visitors.pop()    #remove last visitor, since he or she arrived after closing the park
        return self.visitors
    
    def changestatus(self, x):
        self.status = x

    def savedeptime(self, t):
        self.deptime = t        

class Train:
    
    def __init__(self, number, Cars):
        self.number = number
        self.persons = deque()
        self.capacity = 25*Cars
        

class Event:
    FROGPOND = 0 # constant for arrival at frog pond
    SKUNKHOLLOW = 1 # constant for arrival at Skunk Hollow
    GATORISLAND = 2  # constant for arrival at Gator Island
    RACCOONCORNER = 3  # constant for arrival at Raccoon Corner
    
    

    def __init__(self, typ, time, train):
        self.type = typ
        self.time = time
        self.train = train # train associated with this event
        

    def __lt__(self, other):
        return self.time < other.time
    
class Trains:
    
    def __init__(self, Cars):
        self.persons = deque()
        self.capacity = 25*Cars    

class FES:

    def __init__(self):
        self.events = []

    def add (self, event):
        heapq.heappush (self.events, event)

    def next_( self ):
        return heapq.heappop (self.events)
    
    
class Simulation:
    
    def __init__(self):
        self.NumberofTrainsatStation = 1
        self.FrogPond = Station()     
        self.SkunkHollow = Station()
        self.GatorIsland = Station()
        self.RaccoonCorner = Station()
        
        
    def simulate(self, T):
        #self.FrogPond.visitors = Station.MakeStationQueue(self, 0)
        self.FrogPond.MakeStationQueue(0)
        self.SkunkHollow.MakeStationQueue(1)
        self.GatorIsland.MakeStationQueue(2)
        self.RaccoonCorner.MakeStationQueue(3)
        
        
        res = SimResults()
        queue = len(self.FrogPond.visitors)+len(self.SkunkHollow.visitors)+len(self.GatorIsland.visitors)+len(self.RaccoonCorner.visitors)
        res.registerallvisitors(len(self.FrogPond.visitors), len(self.SkunkHollow.visitors), len(self.GatorIsland.visitors), len(self.RaccoonCorner.visitors))
        t = 0        
        Trainlist = []
        Trainlistevents = []
        fes = FES()    
        for i in range(0,NrTrains):
            Trainlist.append(Train(i, Cars[i]))    
        for i in Trainlist:
            Trainlistevents.append(Event(Event.FROGPOND, t, i))
            t += Warm_up_time/NrTrains
        for i in Trainlistevents:
            fes.add(i)
        
        while queue != 0:                   #long time, so the system will get empty at the end of the day
            e = fes.next_()
            t = e.time
            tr0 = e.train
            Boardingvisitors = []
                
                
            ##FROG POND 
            if e.type == Event.FROGPOND:                               #Check if train arrives at Frogpond
                if t < self.FrogPond.deptime:       #Train is at a station, so station is not available
                    t = self.FrogPond.deptime + Switchtime           #Train can arrive after the departure and has extra switchtime
                    res.registerstoptrain(0)
                Queuelength = 0                                     #Necessary to count waiting visitor
                # print('ik ben op station FG', t)
                for i in self.FrogPond.visitors:                    #Register queue length at time of arrival
                    if i[0] <= t:                                   #Find visitors who have arrived at time t
                        Queuelength += 1                            #Count waiting visitor
                if t > Warm_up_time:
                    res.registerqueuelength(t, Queuelength, 0)      #Register queuelength
                tr0.persons = [i for i in tr0.persons if i != 0]    #visitors leave the train
                freecapacity = tr0.capacity - len(tr0.persons)      #Check free capacity of the train
                if freecapacity > Queuelength:                      #Everybody in queue can board
                    for i in self.FrogPond.visitors:                #Find visitors who have arrived at time t so they can go into the train
                        if i[0] <= t+2:
                            Boardingvisitors.append(i)              #Add those visitors to a list
                    for i in Boardingvisitors:                     
                        res.registerwaitingtime(t-i[0], 0)          #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)      #Check how many people have boarded train
                    for i in range(NrBoardingvisitors):             #Delete peope who have boarded train from visitorstationlist
                        self.FrogPond.visitors.popleft()            
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                    #Visitors in list board
                    
                        
                else:                                                       #Not everybody in queue can board
                    for j in range(freecapacity):                           
                        Boardingvisitors.append(self.FrogPond.visitors[j])  #Add longest waiting visitors to list with people who can board
                    for i in Boardingvisitors:
                        res.registerwaitingtime(t-i[0], 0)                  #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)              #Check how many people have boarded train    
                    for i in range(NrBoardingvisitors):                     #Delete peope who have boarded train from visitorstationlist
                        self.FrogPond.visitors.popleft()
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                            #Visitors in list board
                    for i in self.FrogPond.visitors:                        #Register the number of visitors that can not board
                        if i[0] <= t+2:
                            res.registernotboarders(i[0],0)
                    
                
                x = random.randint(0,100)                                   #Distribution of customers who do not allow a door to close 
                if x <= Delaychance:                                        #Customer don't allow doors to close
                    time = Timetoboard + Delaytime                          
                else:                                                       #Doors close normally
                    time = Timetoboard
                self.FrogPond.savedeptime(t + time)                         #Register the departure time of the train in order to check when a station is empty
                Arrival = Event(Event.SKUNKHOLLOW, t + time + 5, tr0)     #schedule arrival at the next station
                fes.add(Arrival)
                
      
                    


                
                
        
        ##SKUNK HOLLOW 
            if e.type == Event.SKUNKHOLLOW: # if train is at SKUND HOLLOW        
                if t < self.SkunkHollow.deptime:
                    t = self.SkunkHollow.deptime + Switchtime
                    res.registerstoptrain(1)
                Queuelength = 0                                     #Necessary to count waiting visitor
                for i in self.SkunkHollow.visitors:                 #Register queue length at time of arrival
                    if i[0] <= t:                                  #Find visitors who have arrived at time t
                        Queuelength += 1                            #Count waiting visitor
                if t > Warm_up_time:
                    res.registerqueuelength(t, Queuelength, 1)      #Register queuelength
                tr0.persons = [i for i in tr0.persons if i != 1]    #Visitors leave the train
                freecapacity = tr0.capacity - len(tr0.persons)      #Check free capacity of the train
                if freecapacity > Queuelength:                      #Everybody in queue can board
                    for i in self.SkunkHollow.visitors:             #Find visitors who have arrived at time t so they can go into the train
                        if i[0] <= t+2:
                            Boardingvisitors.append(i)              #Add those visitors to a list
                    for i in Boardingvisitors:                     
                        res.registerwaitingtime(t-i[0], 1)          #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)      #Check how many people have boarded train
                    for i in range(NrBoardingvisitors):             #Delete peope who have boarded train from visitorstationlist
                        self.SkunkHollow.visitors.popleft()            
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                    #Visitors in list board
                    
                        
                else:                                                          #Not everybody in queue can board
                    for j in range(freecapacity):                           
                        Boardingvisitors.append(self.SkunkHollow.visitors[j])  #Add longest waiting visitors to list with people who can board
                    for i in Boardingvisitors:
                        res.registerwaitingtime(t-i[0], 1)                     #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)                 #Check how many people have boarded train   
                    for i in range(NrBoardingvisitors):                        #Delete peope who have boarded train from visitorstationlist
                        self.SkunkHollow.visitors.popleft()
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                               #Visitors in list board
                    for i in self.SkunkHollow.visitors:                        #Register the visitors that can not board
                        if i[0] <= t+2:
                            res.registernotboarders(i[0], 1)
                    
                x = random.randint(0,100)                                   #Distribution of customers who do not allow a door to close 
                if x <= Delaychance:                                        #Customer don't allow doors to close
                    time = Timetoboard + Delaytime                          
                else:                                                       #Doors close normally
                    time = Timetoboard
                self.SkunkHollow.savedeptime(t + time)                         #Register the departure time of the train in order to check when a station is empty
                Arrival = Event(Event.GATORISLAND, t + time + 8, tr0)     #schedule arrival at the next station
                fes.add(Arrival)                                  
                
        ##GATOR ISLAND 
        
            if e.type == Event.GATORISLAND:                             # if train is at GATOR ISLAND
                if t < self.GatorIsland.deptime:
                    t = self.GatorIsland.deptime + Switchtime
                    res.registerstoptrain(2)
                Queuelength = 0                                     #Necessary to count waiting visitor
                for i in self.GatorIsland.visitors:                 #Register queue length at time of arrival
                    if i[0] <= t:                                   #Find visitors who have arrived at time t
                        Queuelength += 1                            #Count waiting visitor
                if t > Warm_up_time:
                    res.registerqueuelength(t, Queuelength, 2)      #Register queuelength
                tr0.persons = [i for i in tr0.persons if i != 2]    #Visitors leave the train
                freecapacity = tr0.capacity - len(tr0.persons)      #Check free capacity of the train
                if freecapacity > Queuelength:                      #Everybody in queue can board
                    for i in self.GatorIsland.visitors:             #Find visitors who have arrived at time t so they can go into the train
                        if i[0] <= t+2:
                            Boardingvisitors.append(i)              #Add those visitors to a list
                    for i in Boardingvisitors:                     
                        res.registerwaitingtime(t-i[0], 2)          #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)      #Check how many people have boarded train
                    for i in range(NrBoardingvisitors):             #Delete peope who have boarded train from visitorstationlist
                        self.GatorIsland.visitors.popleft()            
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                    #Visitors in list board
                    
                        
                else:                                                           #Not everybody in queue can board
                    for j in range(freecapacity):                           
                        Boardingvisitors.append(self.GatorIsland.visitors[j])   #Add longest waiting visitors to list with people who can board
                    for i in Boardingvisitors:
                        res.registerwaitingtime(t-i[0], 2)                      #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)                  #Check how many people have boarded train   
                    for i in range(NrBoardingvisitors):                         #Delete peope who have boarded train from visitorstationlist
                        self.GatorIsland.visitors.popleft()
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                                #Visitors in list board
                    for i in self.GatorIsland.visitors:                         #Register the visitors that can not board
                        if i[0] <= t+2:
                            res.registernotboarders(i[0], 2)
                                      
                
                x = random.randint(0,100)                                   #Distribution of customers who do not allow a door to close 
                if x <= Delaychance:                                        #Customer don't allow doors to close
                    time = Timetoboard + Delaytime                          
                else:                                                       #Doors close normally
                    time = Timetoboard
                self.GatorIsland.savedeptime(t + time)                         #Register the departure time of the train in order to check when a station is empty
                Arrival = Event(Event.RACCOONCORNER, t + time + 7, tr0)     #schedule arrival at the next station
                fes.add(Arrival)                                            #Add event to FES
            
        
        
        
        ##RACCOON CORNER 
        
            if e.type == Event.RACCOONCORNER:                                 #If train is at RACOON CORNER
               if t < self.RaccoonCorner.deptime:
                    t = self.GatorIsland.deptime + Switchtime
                    res.registerstoptrain(3)
               Queuelength = 0                                             #Necessary to count waiting visitor
               for i in self.RaccoonCorner.visitors:                       #Register queue length at time of arrival
                    if i[0] <= t:                                         #Find visitors who have arrived at time t
                        Queuelength += 1                                    #Count waiting visitor
               if t > Warm_up_time:
                    res.registerqueuelength(t, Queuelength, 3)      #Register queuelength
               tr0.persons = [i for i in tr0.persons if i != 3]            #Visitors leave the train
               freecapacity = tr0.capacity - len(tr0.persons)              #Check free capacity of the train
               if freecapacity > Queuelength:                              #Everybody in queue can board
                    for i in self.RaccoonCorner.visitors:                   #Find visitors who have arrived at time t so they can go into the train
                        if i[0] <= t+2:
                            Boardingvisitors.append(i)              #Add those visitors to a list
                    for i in Boardingvisitors:                     
                        res.registerwaitingtime(t-i[0], 3)          #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)      #Check how many people have boarded train
                    for i in range(NrBoardingvisitors):             #Delete peope who have boarded train from visitorstationlist
                        self.RaccoonCorner.visitors.popleft()            
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])

                    
                        
               else:                                                               #Not everybody in queue can board
                    for j in range(freecapacity):                           
                        Boardingvisitors.append(self.RaccoonCorner.visitors[j])     #Add longest waiting visitors to list with people who can board
                    for i in Boardingvisitors:
                        res.registerwaitingtime(t-i[0], 3)                          #Register waiting times from visitors who are going to board
                    NrBoardingvisitors = len(Boardingvisitors)                      #Check how many people have boarded train   
                    for i in range(NrBoardingvisitors):                             #Delete peope who have boarded train from visitorstationlist
                        self.RaccoonCorner.visitors.popleft()
                    for i in Boardingvisitors:
                        tr0.persons.append(i[1])                                    #Visitors in list board                       #Visitors in list board
                    
                    for i in self.RaccoonCorner.visitors:                           #Register the visitors that can not board
                        if i[0] <= t+2:
                            res.registernotboarders(i[0], 3)
                    
               x = random.randint(0,100)                                   #Distribution of customers who do not allow a door to close 
               if x <= Delaychance:                                        #Customer don't allow doors to close
                    time = Timetoboard + Delaytime                          
               else:                                                       #Doors close normally
                    time = Timetoboard
               self.GatorIsland.savedeptime(t + time)                         #Register the departure time of the train in order to check when a station is empty
               Arrival = Event(Event.FROGPOND, t + time + 6, tr0)     #schedule arrival at the next station
               fes.add(Arrival)                                            #Add event to FES
               
               
                          
        
            
        
            queue = len(self.FrogPond.visitors)+len(self.SkunkHollow.visitors)+len(self.GatorIsland.visitors)+len(self.RaccoonCorner.visitors)
        
        res.getwaitingTimeResults()
        res.getqueuelengthresults()
        res.getnotboardrate()
        res.getstoptraincount()
        res.getxcount()
        
        
        return res.table


class QUESTIONS:
    
    def __init__(self, Runs):
        self.SM = Simulation()
        self.Runs = Runs
        self.totaltable = pd.DataFrame()
        self.R = SimResults()
        
    def Initialize(self):
        # Make an entire new simulation class for each run
        self.SM=Simulation()
        
    def MultipleSimulations(self, Runs):
        r = 1
        while r <= Runs:
            df = self.SM.simulate(T)
            self.totaltable = self.totaltable.add(df, fill_value = 0)
            
            self.R.Result.append(df.at[3, 'Waiting Time'])
            self.R.ResultMean.append(np.mean(self.R.Result))            
            
            print("At run", r, "time is:", (time.time()-start_time))            
            self.Initialize() #make a new simulation
            r += 1

        self.meantable = self.totaltable.div(Runs)
        
        tbl = self.meantable
        tbl['Interval mean waiting time'] = self.R.CalculateCI(tbl['Waiting Time Squared'], tbl['Waiting Time'], Runs) #calculate CI for waiting time
        tbl['Interval mean Queue length'] = self.R.CalculateCI(tbl['Queue length Squared'], tbl['Queue length'], Runs) #calculate CI for queue length
        tbl['Interval mean Boarder rate'] = self.R.CalculateCI(tbl['Boarder rate Squared'], tbl['Boarder rate'], Runs) #calculate CI for utilization rate
        
        
        filepath1 = Path(os.path.join(os.path.dirname(__file__), 'total.csv')) #assign new file for totaltable csv
        filepath1.parent.mkdir(parents=True, exist_ok=True) #create a directory with parents
        self.totaltable.to_csv(filepath1) #write total.csv
        filepath2 = Path(os.path.join(os.path.dirname(__file__), 'mean.csv')) #assign new file for totaltable csv
        filepath2.parent.mkdir(parents=True, exist_ok=True) #create a directory with parents
        self.meantable.to_csv(filepath2) #write total.csv

        plt.plot(self.R.ResultMean)
        plt.title('Mean waiting time Raccoon Corner')
        plt.xlabel('Runs')
        plt.ylabel('Average waiting time')
        plt.show()
        
        objects = ['x<=25', '25<x<=50', '50<x<=75', 'x>75'] #station numbers
        y_pos = np.arange(len(objects))
        plt.bar(y_pos, tbl['Queue Ratios'], align='center', alpha=0.5) #plot waiting time
        plt.xticks(y_pos, objects)
        plt.xlabel('Distribution')
        plt.title('Distribution of sizes of queuelength')
        plt.show()
        
        
        self.R.Barplotmeans(tbl['Boarder rate'], "Mean Boarder Rate")
        self.R.Barplotmeans(tbl['Waiting Time'], "Mean Waiting Time")
        self.R.Barplotmeans(tbl['Queue length'], "Mean Queue length")

        frogpondboard = [tbl.at[0, 'One wait visitors'], tbl.at[0, 'Two wait visitors'], tbl.at[0, 'Three wait visitors'], tbl.at[0, 'Four wait visitors'], tbl.at[0, 'Five wait visitors'], tbl.at[0, 'Six wait visitors']]
        skunkhollowboard = [tbl.at[1, 'One wait visitors'], tbl.at[1, 'Two wait visitors'], tbl.at[1, 'Three wait visitors'], tbl.at[1, 'Four wait visitors'], tbl.at[1, 'Five wait visitors'], tbl.at[1, 'Six wait visitors']]
        gatorislandboard = [tbl.at[2, 'One wait visitors'], tbl.at[2, 'Two wait visitors'], tbl.at[2, 'Three wait visitors'], tbl.at[2, 'Four wait visitors'], tbl.at[2, 'Five wait visitors'], tbl.at[2, 'Six wait visitors']]
        raccooncorner = [tbl.at[3, 'One wait visitors'], tbl.at[3, 'Two wait visitors'], tbl.at[3, 'Three wait visitors'], tbl.at[3, 'Four wait visitors'], tbl.at[3, 'Five wait visitors'], tbl.at[3, 'Six wait visitors']]
        self.R.Notboarddistribution(frogpondboard, 'Frog Pond')
        self.R.Notboarddistribution(skunkhollowboard, 'Skunk Hollow')
        self.R.Notboarddistribution(gatorislandboard, 'Gator Island')
        self.R.Notboarddistribution(raccooncorner, 'Raccoon Corner')


sim = QUESTIONS (Runs)
sim.MultipleSimulations(Runs)
print("time is:", (time.time()-start_time))

