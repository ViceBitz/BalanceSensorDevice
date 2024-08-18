
/*
Force Sensor Test
*/

int samplingCounter = 0;
const float SAMPLING_RATE = 0.0001;

float sensorBR = 0.0;
float sensorFR = 0.0;
float sensorBL = 0.0;
float sensorFL = 0.0;

float dt = 0.1;
float elapsed = 0.0;
float TIMEOUT = 30;
float SETUP_TIME = 0;

/*
Mode Enums:
0 - Sensor Calibration
1 - Personal Calibration
2 - Balance Analysis
3 - Infinite Run
*/
float MODE = 2;

float analogToVoltage(int num) {
  return num * (5.0 / 1023.0);
}
// the setup function runs once when you press reset or power the board
void setup() {
  // initialize digital pin LED_BUILTIN as an output
  if (MODE < 1) {
    TIMEOUT = 5;
  }
  else if (MODE == 1) {
    TIMEOUT = 15;
  }
  else if (MODE == 2) {
    TIMEOUT = 60;
  }
  else if (MODE == 3) {
    TIMEOUT = 100000;
    SETUP_TIME = 0;
  }
  delay(SETUP_TIME*1000);
  Serial.begin(9600);
}

// the loop function runs over and over again forever
void loop() {
  if (elapsed <= TIMEOUT+0.01) {
    sensorBR = 1/analogToVoltage(analogRead(A0));
    sensorFR = 1/analogToVoltage(analogRead(A1));
    sensorBL = 1/analogToVoltage(analogRead(A2));
    sensorFL = 1/analogToVoltage(analogRead(A3));

    Serial.print(elapsed);
    Serial.print(",");
    Serial.print(sensorBR);
    Serial.print(",");
    Serial.print(sensorFR);
    Serial.print(",");
    Serial.print(sensorBL);
    Serial.print(",");
    Serial.println(sensorFL);
  }
  delay(dt*1000);
  elapsed += dt;
}
