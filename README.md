# Balance Sensor

## Overview

The system quantifies and maps balance over time in patients using a custom-built pressure sensor device (Arduino UNO + Python).  
**Research Paper Draft:** https://www.skylimit.science/balance-sensor

In the elderly population, approximately two-thirds of injury-related deaths are related to accidental falls. We aim to provide a measurable, data-driven approach to balance assessment, helping at-risk patients and caretakers track stability. to help elder.

This sensor system consists of two main parts:
1. Balance Device: collecting balance data by measuring pressure shifts on base of feet with sensor-rigged platforms
2. Signal Processing & Data Analysis: convert voltage signals into force readings, normalize time-series data across different sensitivities, calculate and visualize aggregate balance score  

---
## System Details

#### 📜 Methods

- **Hardware Engineering**: Built a physical balance scale using force-sensitive resistors, voltage divider circuits, and Arduino UNO.
- **Signal Processing**: Collected and processed time-series pressure data through voltage readings.  
- **Device Calibration**: Designed calibration stages to normalize readings for sensor inconsistencies and varying user balance profiles.  
- **Statistical Analysis & Visualization**: Formulated balance scores with custom statistical formula and visualized trends using JMP and Python.  

#### 🎯 Challenges

- **User Variability**: Different stances and weights required personal calibration.  
- **Testing Conditions**: Validated the sensor for eyes-open and eyes-closed scenarios.  
- **Sensor Consistency**: Ensured even pressure distribution using O-rings and a playmat sponge. 

#### 🧪 Testing & Results

- Differentiates between poor and good balance (e.g., eyes closed vs. eyes open, across age groups).  
- Consistent readings on repeated tests for the same person.  
- Clear trends in instability detected.  

---
## Tech Stack

On the hardware side, the circuits feed into an Arduino UNO microcontroller board. Force sensitive resistors read pressure data from the base of both feet, placed directly under the device's platforms with O-ring padding to protect the structure.

We collect time-series pressure data through an INO script ported on the UNO device, which writes to a CSV file. Readings pass through a two-stage normalization process that standardizes sensor irregularities and tunes to user balance profiles. The final score is calculated as the summation of relative shifts in balance at every time step, which we can then plot to visualize stability over time.

TL:DR
- **Hardware**: Arduino UNO, voltage divider circuits, custom sensor platform  
- **Software**: Python (data analysis, visualization, score calculation), INO (embedded programming)

