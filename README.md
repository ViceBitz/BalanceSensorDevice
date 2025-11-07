# Balance Sensor

The system quantifies and maps balance over time in patients using a custom-built pressure sensor device (Arduino UNO + Python).  
✏️ Research Paper Draft: https://www.skylimit.science/balance-sensor

### Inspiration
In the elderly population, approximately two-thirds of injury-related deaths are related to accidental falls. We aim to provide a measurable, data-driven approach to balance assessment, helping at-risk patients and caretakers track stability. to help elder.  

---

### 📜 Methods

- **Hardware Engineering**: Built a physical balance scale using force-sensitive resistors, voltage divider circuits, and Arduino UNO.
- **Signal Processing**: Collected and processed time-series pressure data through voltage readings.  
- **Device Calibration**: Designed calibration stages to normalize readings for sensor inconsistencies and varying user balance profiles.  
- **Statistical Analysis & Visualization**: Formulated balance scores with custom statistical formula and visualized trends using JMP and Python.  

### 🎯 Challenges

- **User Variability**: Different stances and weights required personal calibration.  
- **Testing Conditions**: Validated the sensor for eyes-open and eyes-closed scenarios.  
- **Sensor Consistency**: Ensured even pressure distribution using O-rings and a playmat sponge. 

### 🧪 Testing & Results

- Differentiates between poor and good balance (e.g., eyes closed vs. eyes open, across age groups).  
- Consistent readings on repeated tests for the same person.  
- Clear trends in instability detected.  

---

### 🛠️ Tech Stack

- **Hardware**: Arduino UNO, voltage divider circuits, custom sensor platform  
- **Software**: Python (data analysis, visualization, score calculation), INO (embedded programming)
