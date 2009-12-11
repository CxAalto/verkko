"""
Module for analysing events in mobile phone networks.


Loading events:
numberOfUsers=3786611
numberOfEvents=74368195
events=PhoneEventsContainer("/proj/net_scratch/Mikko/phone_dynamics/events_lccAll_reversed_false_timesorted.npy",numberOfEvents,numberOfUsers,reversed=False,format="numpy",sortOrder=("time",))

Loading events and added reversed events:
events=PhoneEventsContainer("/proj/net_scratch/Mikko/phone_dynamics/lcCalls_reverse_users_time.npy",numberOfEvents,numberOfUsers,reversed=True,format="numpy",sortOrder=("fr","time"))



"""

import os
from sys import stdout
import numpy
from time import *
from collections import deque

class PhoneEvent(object):
    defaultStartTime = mktime(strptime("20070101 000000", "%Y%m%d %H%M%S"))

    def __init__(self,line=None,startTime=None,record=None,
                 format="orig",originalIndex=None):
        startTime = (startTime or self.defaultStartTime)
        
        if line != None:
            self.parseEvent(line, startTime, format)
            assert(originalIndex != None)
            self.originalIndex = originalIndex

        if record != None:
            # Maybe this data could be stored to record in this class also?
            self.time = record['time']            
            self.fr = record['fr']
            self.to = record['to']
            self.duration = record['duration']
            self.call = record['call']
            self.reversed = record['reversed']
            self.originalIndex = record['originalIndex']

    def __eq__(self,other):
        return self.originalIndex == other.originalIndex
        #check only the original index
        #return self.fr==other.fr and self.to == other.to and self.time==other.time
    
    def parseEvent(self,line,startTime,format):
        """Set properties according to a line from event file."""
        fields = line.split()

        if format == "orig":
            # Read event time
            self.time = mktime(strptime(fields[0]+' '+fields[1],
                                        "%Y%m%d %H%M%S")) - startTime
            self.fr = int(fields[2])
            self.to = int(fields[4])
            self.duration = int(fields[5])
            self.call = (fields[3] == "2")
            self.reversed = False

        elif format=="sec":
            self.time = int(fields[0])
            self.fr = int(fields[1])
            self.to = int(fields[3])
            self.duration = int(fields[4])
            self.call = (fields[2] == "2")
            self.reversed = False

    def getReversed(self):
        newEvent=PhoneEvent()

        #These values are the same:
        newEvent.time=self.time
        newEvent.duration=self.duration
        newEvent.call=self.call
        newEvent.originalIndex=self.originalIndex

        #These values are changed:
        newEvent.fr=self.to
        newEvent.to=self.fr
        newEvent.reversed = (not self.reversed)
    
        return newEvent

    def  __str__(self):
        v = (self.time, self.fr, self.to, self.duration,
             int(self.call), int(self.reversed))
        return ", ".join(map(str, v))
        

    def __hash__(self):
        return self.originalIndex.__hash__()
        #return str(self).__hash__()

class PhoneEvents(object):
    """A virtual class for handling and parsing phone events."""
    
    def __init__(self, startTime=None, sortOrder=()):
        self.startTime = (startTime or PhoneEvent.defaultStartTime)
        self.sortOrder = sortOrder
        self.indexEvents = False

    def _userEvents(self,firstEvent,iterator,lastEventContainer):
        #thisUser=firstEvent.fr
        yield firstEvent
        nextEvent=iterator.next()
        lastEventContainer[0]=nextEvent
        while nextEvent.fr==nextEvent.fr:
            yield nextEvent
            nextEvent=iterator.next()
            lastEventContainer[0]=nextEvent
        print lastEventContainer

    def getUsers_new(self):

        if len(self.sortOrder) < 1 or self.sortOrder[0] != "fr":
            raise Exception("Events must be sorted by 'fr'.")
        thisUser = None
        userEvents = []
        lastEventContainer = [None]
        try:
            iterator = self.__iter__()
            firstEvent = iterator.next()
            yield self._userEvents(firstEvent,iterator,lastEventContainer)
            while True:
                event = iterator.next()
                yield self._userEvents(lastEventContainer[0],
                                       iterator, lastEventContainer)
        except StopIteration:
            pass

    def getUsers(self):
        if len(self.sortOrder)<1 or self.sortOrder[0]!="fr":
            raise Exception("Events must be sorted by 'fr'.")
        thisUser=None
        userEvents=[]
        for event in self:
            if thisUser==None:
                thisUser=event.fr

            if event.fr==thisUser:
                userEvents.append(event)
            else:
                yield userEvents
                userEvents=[event]
                thisUser=event.fr

    

    def getTriggeredEvents_tree(self, delta, allowReturn=True):
        """This generator returns all possible triggered events which
        have time delta between them, and no other triggering event between
        them.
        """
        if ( len(self.sortOrder) < 2 or 
             self.sortOrder[0] != "fr" or 
             self.sortOrder[1] != "time" ):
            raise Exception("Events are not properly sorted.")
        for userEvents in self.getUsers():
            lastReversedEvent=None
            for event in userEvents:
                if event.reversed:
                    lastReversedEvent=event
                elif ( lastReversedEvent != None and
                       (event.time - lastReversedEvent.time -
                        lastReversedEvent.duration) < delta ):
                    if allowReturn or event.to!=lastReversedEvent.to:
                        yield (lastReversedEvent.getReversed(), event)
        #remove sms

    def getTriggeredEvents_net(self, delta, allowReturn=True,allowOutgoingCalls=True):
        """This generator returns all possible triggered events which
        have time delta between them.

        TODO: looping time
        """
        if ( len(self.sortOrder) < 2 or 
             self.sortOrder[0] != "fr" or 
             self.sortOrder[1] != "time" ):
            raise Exception("Events are not properly sorted.")
        for userEvents in self.getUsers():
            #lastReversedEvents=[] #you should use queue instead of list. this is very slow
            lastReversedEvents=deque()
            for event in userEvents:
                if event.reversed:
                    lastReversedEvents.append(event)
                else:
                    #first remove all too old events
                    while len(lastReversedEvents)!=0 and  int(event.time)-int(lastReversedEvents[0].time)-int(lastReversedEvents[0].duration) > delta:
                        lastReversedEvents.popleft()
                    #then add link from the sufficiently new events to the current one
                    for oldEvent in lastReversedEvents:
                        if allowReturn or event.to!=oldEvent.to:
                            timeBetween=int(event.time)-int(oldEvent.time)-int(oldEvent.duration)
                            yield (oldEvent.getReversed(), event,timeBetween)

                    #finally add the event to the queue if it is a call and
                    #triggering is allowed for outgoing calls also
                    if allowOutgoingCalls and event.call:
                        lastReversedEvents.append(event.getReversed())

    def getInterEventTimes(self,loopTime=False):
        """
        Returns an array of inter-event times. 
        The time at index i of the array corresponds to the difference
        in time from the event i and the last event between the two 
        users. That is, if event i is a call between users A and B, the
        inter-event time with index i would be IE_i = t_i - t_j, where
        t_i is the time of the event i and t_j is the time of the event j,
        which is the last event between A and B before event i.
        
        Parameters
        ----------
        loopTime : bool
            If True, periodic boundary conditions for time is used,
            which means that the time between the first and the last
            event in the data is set to zero. If False, the leading
            events of the first events are considered missing, and a
            negative value for inter-event time is assigned to them.
        
        """
        def getEvents(self):
            thisEvent=None
            events=[]
            for event in self:
                if thisEvent==None:
                    thisEvent=(event.fr,event.to)

                if (event.fr,event.to)==thisEvent:
                    events.append(event)
                else:
                    yield events
                    events=[event]
                thisEvent=(event.fr,event.to)


        if ( len(self.sortOrder) < 3 or 
             self.sortOrder[0] != "fr" or 
             self.sortOrder[1] != "to" or 
             self.sortOrder[2] != "time" ):
            raise Exception("Events are not properly sorted.")
        
        if not self.reversed:
            raise Exception("The data must contain reversed events.")

        if loopTime:
            dataDuration=max(self.eventData.time)-min(self.eventData.time)

        ieTimes=numpy.zeros(self.numberOfEvents/2,dtype="int32")
        #for i in range(len(ieTimes)):
        #    ieTimes[i]=-2

        for events in getEvents(self):
            if events[0].fr<events[0].to:
                for i,event in enumerate(events):
                    thisTime=event.time
                    previousTime=events[i-1].time
                    if i==0:
                        if loopTime:
                            deltaTime=thisTime-previousTime+dataDuration
                        else:
                            deltaTime=-1
                    else:
                        deltaTime=thisTime-previousTime
                    ieTimes[event.originalIndex]=deltaTime
        
        return ieTimes


class PhoneEventsStatic(PhoneEvents):
    pass

class PhoneEventsContainer(PhoneEvents):
    """A container for phone events.

    All events are kept in the memory for fast random access.
    """
    
    def __init__(self, inputFileName, numberOfEvents=None, numberOfUsers=None,
                 reversed=False,startTime=None,verbose=True,format=None,
                 sortOrder=()):
        # Initialize parent class. Note that if startTime is None,
        # PhoneEvents uses the default start time (defined in
        # PhoneEvents).
        PhoneEvents.__init__(self, startTime = startTime, sortOrder=sortOrder)
        self.numberOfUsers=numberOfUsers # Used nowhere? (LK 4.8.2008)

        self.reversed = reversed

        self.iterateCalls = True # What are
        self.iterateSMS = True   #  these?

        # Deduce file format if not given.
        if format is None:
            if (isinstance(inputFileName, str) 
                and inputFileName.split('.')[-1] == 'npy'):
                format = 'numpy'
            else:
                format = 'orig'

        if format != "numpy":
            if isinstance(inputFileName, str):
                inputFileName = [inputFileName]

            # If the number of events is not specified, use the number
            # of rows in the input file.
            if not numberOfEvents:
                numberOfEvents = 0
                for fn in inputFileName:
                    numberOfEvents += int(os.popen("wc -l %s"%fn).next().split()[0])
                print ("Total of %d events found in %d file%s." % 
                       (numberOfEvents, len(inputFileName),
                        ("s" if len(inputFileName) > 1 else "")))

            if reversed:
                self.numberOfEvents = numberOfEvents*2
            else:
                self.numberOfEvents = numberOfEvents

            # Initialize record array for event data.
            column_formats = 'uint32,uint32,uint32,uint16,bool,bool,uint32'
            column_names = 'fr,to,time,duration,call,reversed,originalIndex'
            self.eventData = numpy.recarray(self.numberOfEvents,
                                            formats=column_formats,
                                            names=column_names)

            lineNumber = 0
            for fn in inputFileName:
                with open(fn, 'r') as f:
                    for line in f:
                        thisEvent = PhoneEvent(line, format=format,
                                               originalIndex=lineNumber)
                        if reversed:
                            self._addEvent(thisEvent, lineNumber*2)
                            self._addEvent(thisEvent.getReversed(), lineNumber*2+1)
                        else:
                            self._addEvent(thisEvent,lineNumber)

                        if verbose and lineNumber % 1000000 == 0:
                            print lineNumber

                        lineNumber += 1
        else:
            # Read in a previously saved numpy.recarray object.
            self.eventData = numpy.load(inputFileName)
            self.numberOfEvents = len(self.eventData)
            
    def _addEvent(self, event, eventIndex):
            self.eventData['fr'][eventIndex] = event.fr
            self.eventData['to'][eventIndex] = event.to
            self.eventData['call'][eventIndex] = event.call
            self.eventData['duration'][eventIndex] = event.duration
            self.eventData['time'][eventIndex] = event.time
            self.eventData['reversed'][eventIndex] = event.reversed
            self.eventData['originalIndex'][eventIndex] = event.originalIndex

    def sort(self,field):
        self.eventData.sort(order=field)
        if field.__class__ == tuple:
            self.sortOrder=field
        else:
            self.sortOrder=(field,)

    def shuffle(self, field=None):
        """Shuffle either full data or one column.

        This method works differently depending on whether an input
        parameter `field` is given:
           If `field` is not given or is None, the ordering of events
           is randomized.

           If `field` is given and it corresponds to one of the data
           fields (for example 'time'), the ordering of `field` is
           preserved while the data on other columns is shuffled. This
           is useful if for instance the original events are already
           sorted by time, and you want to shuffle the time stamps of
           the events.
        """
        if self.reversed:
            raise Exception("Can't shuffle reversed events.")
        if field == None:
            # Shuffle everything
            numpy.random.shuffle(self.eventData)
        else:
            # Keep field in original order, shuffle everything else.
            orig_col = self.eventData[field].copy()
            numpy.random.shuffle(self.eventData)
            self.eventData[field] = orig_col
            

    def __iter__(self):
        for index, eventRecord in enumerate(self.eventData):
            event = PhoneEvent(record=eventRecord)
            if event.call:
                if self.iterateCalls:
                    yield event
            else:
                if self.iterateSMS:
                    yield event

    def saveData(self,fileName):
        numpy.save(fileName, self.eventData)

    def __getitem__(self,index):
        return PhoneEvent(record = self.eventData[index])

    def __len__(self):
        return len(self.eventData)

