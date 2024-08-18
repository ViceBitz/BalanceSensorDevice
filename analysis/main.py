import csv
import numpy as np
import matplotlib.pyplot as plt
import math

"""
Main driver of the pressure sensor project. Calibrates sensors in two stages: (1) sensor, (2) personal,
and then runs the real-time evaluation and calculates the balance score. All data includes a timestamp
as well as the 1/voltage reading.

*Force is inversely proportional to voltage, hence using 1/voltage linearizes

@author Victor Gong
@version 8/18/2024
"""

""" *All sensor orderings are (BR, FR, BL, FL), readings start with timestamp """
SENSOR_COUNT = 4
SUBPERIOD_LENGTH = 5 #seconds

#Calibration files
sensorCaliFileName = "data/s_cali/sensor_calibrate.csv" #Sensor profile, Format: title line, (# of weights), (# of readings, weight), [each reading...], ...
personalCaliFileName = "data/p_cali/A.csv" #Personal balance profile, Format: title line, (# of readings), [each reading...]

#Evaluation files
evalFileName = "data/evaluation.csv" #Data to evaluate, Format: title line, (# of readings), [each reading...]

#Output files
saveFileName = "data/eval_results/A.csv" #Results save file, Format: (# of readings), [each reading...]

"""
---------------------------------------
            FILE MANAGEMENT
---------------------------------------
"""

#Loads a raw reading into the evaluation file
def loadEvaluation(rawFileName):
   readingsData = []
   #Read from raw file
   with open(rawFileName, "r") as f:
      for line in f.readlines():
         readingsData.append(line.split(","))
   #Write to evaluation file
   with open(evalFileName, "w") as csvF:
      csvWriter = csv.writer(csvF)
      csvWriter.writerow(["T","BR","FR","BL","FL"])
      csvWriter.writerow([len(readingsData)])
      for data in readingsData:
         csvWriter.writerow(data)

"""
---------------------------------------
         SENSOR CALIBRATION
---------------------------------------
(1) 4 sensors have different sensitivities due to physical differences
   (i) Results in different [F vs. 1/V] lines, with different slopes and offsets

(2) Calibrate with 5 different weights placed on top of contraption, and line fit
the data for an approximate y = mx + b.

(3) Normalize data by saving the slope (m) and offset (b), and using the
transformation y = (y0 - b) / m (changes all lines into slope = 1, offset = 0)

**Units of weights don't matter because relative throughout
"""
s_cali = [] #Format: [(slope, offset),...]

def calibrateSensors(plotData=False):
   global s_cali
   with open(sensorCaliFileName, "r") as csvF:
      csvReader = csv.reader(csvF)
      csvContent = [line for line in csvReader]

      #Get points from data collected during sensor calibration stage
      sensorPoints = [] #Format: [[(weight, reading),...],...]
      for i in range(SENSOR_COUNT): sensorPoints.append([])

      it = 1
      numWeights = int(csvContent[it][0]); it+=1
      for _ in range(numWeights): #Loop through all data per weight
         numReadings = int(csvContent[it][0])
         weight = float(csvContent[it][1]); it+=1
         readingAvg = np.zeros(SENSOR_COUNT) #Avgs for each sensor

         for _ in range(numReadings): #Loop through all readings for that weight
            t = csvContent[it][0]
            values = csvContent[it][1:]
            for i in range(len(values)): values[i] = float(values[i])

            readingAvg += np.array(values)
            it+=1 
         readingAvg /= numReadings #Get the average reading value
         for i in range(SENSOR_COUNT): sensorPoints[i].append((weight, readingAvg[i])) #Add point to sensor points

      #Calculate slope (m) and offset (b) of best fit lines for each sensor
      for i in range(SENSOR_COUNT):
         weights = [] #X
         readings = [] #Y
         for w,r in sensorPoints[i]:
            weights.append(w); readings.append(r)
         
         m, b = np.polyfit(np.array(weights), np.array(readings), 1)
         s_cali.append((m, b)) #Add to sensor calibration data

         #Visualize plot if plotData is True
         if plotData:
            plt.plot(weights, readings, "o", color=plot_colors[i])
            lineFit = np.linspace(weights[0],weights[len(weights)-1],10)
            plt.plot(lineFit, m*lineFit+b, "-")
            plt.title("Force Sensitive Resistor Calibration Lines", fontsize=16); plt.xlabel("Total Weight (lb)", fontsize=14); plt.ylabel("1/Resistor Voltage (1/V)", fontsize=14)
            plt.xticks(fontsize=10); plt.yticks(fontsize=10)
            plt.ylim(0, 0.8)
      if plotData: plt.show()



"""
---------------------------------------
         PERSONAL CALIBRATION
---------------------------------------
(1) Different individuals may have different "balanced states", or varying
sensor readings during an ideal balance depending on posture/weight

(2) Calibrate by prompting subject to stand on sensor device for 30 seconds while
holding onto something to balance themselves as perfectly as possible

(3) Normalize data by taking readings as the fractional change of calibrated value,
or in other words, y' = (y - p) / p, where p is the average sensor value from the personal
calibration stage
"""
p_cali = np.zeros(SENSOR_COUNT)

def calibratePersonal():
   global p_cali
   with open(personalCaliFileName, "r") as csvF:
      csvReader = csv.reader(csvF)
      csvContent = [line for line in csvReader]

      #Get average values from data collected during personal calibration stage
      it = 1
      numReadings = int(csvContent[it][0]); it+=1

      for _ in range(numReadings): #Loop through all readings
         t = csvContent[it][0]
         values = csvContent[it][1:]
         for i in range(len(values)): values[i] = s_norm(float(values[i]), i)
         p_cali = p_cali + np.array(values)
         it+=1
      p_cali /= numReadings #Get the average reading value

"""
---------------------------------------
         CALIBRATION METHODS
---------------------------------------
"""
#Method to sensor-normalize any value, with y = (y0 - b) / m
def s_norm(val, sensorIndex):
   return (val - s_cali[sensorIndex][1]) / s_cali[sensorIndex][0]

#Method to personal-normalize (fractional change) any value, with y' = (y - p) / p
def p_norm(val, sensorIndex):
   return (val - p_cali[sensorIndex]) / p_cali[sensorIndex]

#Method to normalize any value, applying s_norm then p_norm
def norm_data(val, sensorIndex):
   return p_norm(s_norm(val, sensorIndex), sensorIndex)


"""
---------------------------------------
               EVALUATION
---------------------------------------
(1) 1 minute snapshot of balance data, cut into subperiods

(2) Get each sensor readings for every dt (unit baud rate):
   (i) Apply sensor normalization on values
   (ii) Apply personal normalization (to calculate fractional change)
      (b) Fractions are necessary because different people have different weights,
      linear methods (addition, subtraction) won't account for this difference
      (see personal normalization)

(4) ||F|| = sqrt(f(1)^2 + f(2)^2 + f(3)^2 + f(4)^2 + ...), and calculate ∫||F|| dt for each subperiod

(5) Plot balance score per subperiod (every 'SUBPERIOD_LENGTH' seconds)
"""

#Calculates balance score per subperiod with normalized sensor readings
def evaluateBalance(printAverage=True):
   balanceScores = [] #Scores per subperiod, Format: [score, ...] per subperiod
   allReadings = [] #Normalized sensor readings, Format: [(time, s1, s2, s3, s4)] *See top of file for ordering info
   with open(evalFileName, "r") as csvF:
      csvReader = csv.reader(csvF)
      csvContent = [line for line in csvReader]

      #Go through every reading
      it = 1
      numReadings = int(csvContent[it][0]); it+=1

      prevT = -1
      currentPeriod = 0.0
      periodIntegral = 0.0

      for n in range(numReadings): #Loop through all readings
         t = float(csvContent[it][0])
         allReadings.append(csvContent[it])
         it+=1

         #Normalize values and convert to float
         for i in range(1,len(allReadings[n])):
            allReadings[n][i] = norm_data(float(allReadings[n][i]), i-1)

         if prevT != -1: #Skip first frame
            #Calculate net magnitude (sqrt(a^2+b^2+c^2...)) and add to integral (right Riemann sum)
            mF = 0.0
            for i in range(SENSOR_COUNT): mF += allReadings[n][1:][i]**2
            mF = math.sqrt(mF)
            periodIntegral += mF * (t - prevT)

         if t%SUBPERIOD_LENGTH==0 and t != 0: #Check if entered next period
            balanceScores.append(periodIntegral) # ∫|F| dt
            prevT = -1; currentPeriod+=1; periodIntegral = 0.0 #Reset vars

         prevT = t
   
   times = []
   for i in range(len(balanceScores)):
      times.append(SUBPERIOD_LENGTH*(i+1))

   #Write data and balance scores to save file
   with open(saveFileName, "w") as csvF:
      csvWriter = csv.writer(csvF)

      numPeriods = len(times)
      csvWriter.writerow([numPeriods]) #Write balance scores per subperiod
      for i in range(numPeriods):
         csvWriter.writerow([times[i], balanceScores[i]])

      numReadings = len(allReadings)
      csvWriter.writerow([numReadings]) #Write raw sensor readings
      for n in range(numReadings):
         csvWriter.writerow(allReadings[n])

      

   #Print average
   if printAverage:
      scoreAvg = 0.0
      for score in balanceScores:
         scoreAvg += score
      print(scoreAvg/len(balanceScores))

"""
---------------------------------------
            VISUALIZATION
---------------------------------------
(1) Reading data from evaluation results files

(2) Plotting data with Python's MatPlotLib
"""
plot_colors = [
   "#1f77b4",
   "#ff7f0e",
   "#2ca02e",
   "#d62728",
   "#9467bd",
   "#8c564b",
   "#e377c2",
   "#7f7f7f",
   "#bcbd22",
   "#17becf"
   ]

#Visualize normalized sensor readings (fraction change), *1 file only due to number of lines
def plotSensorReadings(saveFileName):
   times = []; readings = []
   for i in range(SENSOR_COUNT): readings.append([]) #Initialize readings list for each sensor

   #Read timestamp and readings line by line
   with open(saveFileName, "r") as csvF: 
      csvReader = csv.reader(csvF)
      csvContent = [line for line in csvReader]
      
      it = 0; numPeriods = int(csvContent[it][0]); it+=1+numPeriods
      numReadings = int(csvContent[it][0]); it+=1
      for _ in range(numReadings):
         times.append(float(csvContent[it][0]))
         for i in range(SENSOR_COUNT):
            readings[i].append(float(csvContent[it][i+1]))
         it+=1

   for i in range(SENSOR_COUNT): #Plot
      plt.plot(times, readings[i], color=plot_colors[i])
   plt.show()

#Visualize cumulative sensor readings (Euclidean norm)
def plotCumulativeReadings(saveFileNameList, plot_colors=plot_colors):
   colorIndex = 0
   maxReading = 0.0
   for saveFileName in saveFileNameList:
      times = []; readings = []
      #Read timestamp and readings line by line
      with open(saveFileName, "r") as csvF: 
         csvReader = csv.reader(csvF)
         csvContent = [line for line in csvReader]
         
         it = 0; numPeriods = int(csvContent[it][0]); it+=1+numPeriods
         numReadings = int(csvContent[it][0]); it+=1
         for _ in range(numReadings):
            times.append(float(csvContent[it][0]))
            mF = 0.0 #Recalculate Euclidean norm
            for i in range(SENSOR_COUNT):
               mF += float(csvContent[it][i+1])**2
            readings.append(math.sqrt(mF))
            it+=1
      #Plot data
      plt.plot(times, readings, color=plot_colors[colorIndex])
      colorIndex+=1
      maxReading = max(max(readings), maxReading) #Keep track of max reading for graph scaling
   #Show plot
   plt.title("Consummate Deviation From Personal Balance Profile (Euclidean Dist.)"); plt.xlabel("Time (s)"); plt.ylabel("Combined Deviation")
   plt.xlim(0,times[len(times)-1]); plt.ylim(0,maxReading+0.2)
   plt.grid()
   plt.show()

#Visualize balance scores over the period
def plotBalanceScores(saveFileNameList, plotAverage=True, printAverage=True, plot_colors=plot_colors):
   colorIndex = 0
   maxScore = 0.0
   for saveFileName in saveFileNameList:
      times = []; scores = []
      #Read timestamp and readings line by line
      with open(saveFileName, "r") as csvF: 
         csvReader = csv.reader(csvF)
         csvContent = [line for line in csvReader]
         
         it = 0
         numPeriods = int(csvContent[it][0]); it+=1
         for _ in range(numPeriods):
            times.append(float(csvContent[it][0]))
            scores.append(float(csvContent[it][1]))
            it+=1
      #Plot data
      plt.plot(times, scores, "-o", color=plot_colors[colorIndex])
      #Calculate average and plot
      if plotAverage:
         avgScore = 0.0
         for s in scores: avgScore += s
         avgScore /= len(scores)
         plt.plot([0.0, times[len(times)-1]],[avgScore, avgScore],'--',color=plot_colors[colorIndex])
         if printAverage: print(avgScore)
      colorIndex+=1
      maxScore = max(max(scores), maxScore) #Keep track of max reading for graph scaling

   #Show plot
   plt.title("Balance Test Results"); plt.xlabel("Time (s)"); plt.ylabel("Balance Score")
   plt.xlim(0,times[len(times)-1]); plt.ylim(0,maxScore+0.25)
   plt.grid()
   plt.show()
   
"""
---------------------------------------
         EXPERIMENTAL EVALUATION
---------------------------------------
(1) Instead of the integral method used in main evaluation, apply an FFT
to the fractional change data

(2) Use the frequencies and amplitudes from the FFT to calculate balance score
"""
def evaluateBalanceFFT(plotData=True):
   times = []; fracs = []
   for i in range(SENSOR_COUNT): fracs.append([]) #Initialize readings list for each sensor

   with open(saveFileName, "r") as csvF: #Read timestamp and readings line by line
      csvReader = csv.reader(csvF)
      csvContent = [line for line in csvReader]
      
      it = 0; numPeriods = int(csvContent[it][0]); it+=1+numPeriods
      numReadings = int(csvContent[it][0]); it+=1
      for _ in range(numReadings):
         times.append(float(csvContent[it][0]))

         #Use difference of fractional changes for FFT
         BR = float(csvContent[it][1]); FR = float(csvContent[it][2]); BL = float(csvContent[it][3]); FL = float(csvContent[it][4])
         VR = FR - BR; VL = FL - BL; HF = FR - FL; HB = BR - BL

         for k in range(SENSOR_COUNT):
            fracs[k].append([VR,VL,HF,HB][k])

         it+=1
   #Perform FFT
   Fs = 10 #Sampling frequency (10 samples / s)
   N = len(times) #Number of samples
   f0 = Fs / N #Signal frequency

   FFT = []; FFT_mag_plot = []
   for i in range(SENSOR_COUNT):
      FFT.append(np.fft.fft(fracs[i]))
      FFT_mag_plot.append(2 * (np.abs(FFT[i]) / N)[0:int(N/2+1)])
      FFT_mag_plot[i][0] = FFT_mag_plot[i][0] / 2

   
   if plotData: #Plot FFT
      freq = np.linspace(0, (N-1) * (Fs / N), N) #Frequency steps
      fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
      for i in range(SENSOR_COUNT):
         ax1.plot(times, fracs[i])
         ax2.plot(freq[0:int(N/2+1)], FFT_mag_plot[i], '-'); ax2.set_xlim([0,1]); ax2.set_ylim([0,2])
      plt.show()

#=========Analysis=========
calibrateSensors(plotData=False)

#Process all patient data
profiles = [
   ("data/raw_readings/evaluation_A","data/p_cali/A.csv","data/eval_results/A.csv"),
   ("data/raw_readings/evaluation_B","data/p_cali/B.csv","data/eval_results/B.csv"),
   ("data/raw_readings/evaluation_C","data/p_cali/C.csv","data/eval_results/C.csv"),
   ("data/raw_readings/evaluation_D","data/p_cali/D.csv","data/eval_results/D.csv"),
   ("data/raw_readings/evaluation_E","data/p_cali/E.csv","data/eval_results/E.csv"),
   ("data/raw_readings/evaluation_F","data/p_cali/F.csv","data/eval_results/F.csv"),
   ("data/raw_readings/evaluation_G","data/p_cali/G.csv","data/eval_results/G.csv"),
   ("data/raw_readings/evaluation_H","data/p_cali/H.csv","data/eval_results/H.csv")
]
for rawEvalFile, caliFile, resFile in profiles:
   loadEvaluation(rawEvalFile)
   personalCaliFileName = caliFile
   saveFileName = resFile

   calibratePersonal()
   evaluateBalance(printAverage=False)

#evaluateBalanceFFT()

#=========Visualization=========

resultsFileList = [
   "data/eval_results/A.csv",
   "data/eval_results/B.csv",
   "data/eval_results/C.csv",
   "data/eval_results/D.csv",
   "data/eval_results/E.csv",
   "data/eval_results/F.csv",
   "data/eval_results/G.csv",
   "data/eval_results/H.csv"
]

#plotSensorReadings()
plotCumulativeReadings(resultsFileList)
plotBalanceScores(resultsFileList, printAverage=True)

