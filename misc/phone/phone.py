"""
Module for analysing events in mobile phone networks.


Loading events:
numberOfUsers=3786611
numberOfEvents=74368195
events=PhoneEventsContainer("/proj/net_scratch/Mikko/phone_dynamics/events_lccAll_reversed_false_timesorted.npy",numberOfEvents,numberOfUsers,reversed=False,format="numpy",sortOrder=("time",))

Loading events and added reversed events:
events=PhoneEventsContainer("/proj/net_scratch/Mikko/phone_dynamics/lcCalls_reverse_users_time.npy",numberOfEvents,numberOfUsers,reversed=True,format="numpy",sortOrder=("fr","time"))



"""

import numpy
from time import *

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
        # Reduce is slow and ugly. join is better.
        #return reduce(lambda x,y:x+","+y,
        #              map(str,
        #                  (self.time,self.fr,self.to,self.duration,
        #                   int(self.call),int(self.reversed))))
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
            raise Exception("Events are not properly sorted.")
        #thisUser = None
        #userEvents = []
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
            raise Exception("Events are not properly sorted.")
        thisUser=None
        userEvents=[]
        for event in self:
            if thisUser==None:
                thisUser=event.fr

            if event.fr==thisUser:
                userEvents.append(event)
            else:
                yield userEvents
                userEvents=[]
                thisUser=event.fr



    def getTriggeredEvents(self, delta, allowReturn=True):
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

class PhoneEventsStatic(PhoneEvents):
    pass

class PhoneEventsContainer(PhoneEvents):
    """A container for phone events.

    All events are kept in the memory for fast random access.
    """
    
    def __init__(self, inputFileName, numberOfEvents=None, numberOfUsers=None,
                 reversed=False,startTime=None,verbose=True,format="orig",
                 sortOrder=()):

        # Initialize parent class. Note that if startTime is None,
        # PhoneEvents uses the default start time (defined in
        # PhoneEvents).
        PhoneEvents.__init__(self, startTime = startTime, sortOrder=sortOrder)
        self.numberOfUsers=numberOfUsers # Used nowhere? (LK 4.8.2008)

        self.reversed = reversed

        self.iterateCalls = True # What are
        self.iterateSMS = True   #  these?

        if format != "numpy":
            # The number of events must only be specified when the
            # data is read from a text file. If a ready-saved
            # numpy.recarray is used instead, the number of events can
            # be read from it.
            if not numberOfEvents:
                raise Exception("Number of events must be "
                                "specified with text data.")
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

            inputFile = open(inputFileName,'r')
            for lineNumber, line in enumerate(inputFile):
                thisEvent = PhoneEvent(line, format=format,
                                       originalIndex=lineNumber)
                if reversed:
                    self._addEvent(thisEvent, lineNumber*2)
                    self._addEvent(thisEvent.getReversed(), lineNumber*2+1)
                else:
                    self._addEvent(thisEvent,lineNumber)

                if verbose and lineNumber % 1000000 == 0:
                    print lineNumber
            inputFile.close()

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

    def shuffle(self):
        if self.reversed:
            raise Exception("Can't shuffle reversed events.")
        numpy.random.shuffle(self.eventData)

    def __iter__(self):
        for index, eventRecord in enumerate(self.eventData):
            event = PhoneEvent(record=eventRecord)
            if event.call:
                if self.iterateCalls:
                    yield event
            else:
                if self.iterateSMS:
                    yield event

    def saveData(self,filename):
        numpy.save(filename, self.eventData)

    def __get__(self,index):
        return PhoneEvent(record = self.eventData[index])

    def shuffled_iter(self, field):
        """Iterate shuffled data, keeping `field` ordered.

        This is useful for example when the data is ordered by field
        'time', and we want to shuffle the time stamps of the
        events. Note that this function doesn't alter the ordering of
        the data structure; it is only iterated in a shuffled order.
        """
        # Create shuffled ordering
        rand_order = range(self.numberOfEvents)
        numpy.random.shuffle(rand_order)
        
        for i, ri in enumerate(rand_order):
            rand_record = self.eventData[ri].copy()
            rand_record[field] = self.eventData[i][field]
            yield PhoneEvent(record = rand_record)
