#define IN1 8
#define IN2 9
#define IN3 10
#define IN4 11

// Half-step sequence for 28BYJ-48 (8 steps)
const int stepPattern[8][4] = {
  {1, 0, 0, 0}, {1, 1, 0, 0}, {0, 1, 0, 0}, {0, 1, 1, 0},
  {0, 0, 1, 0}, {0, 0, 1, 1}, {0, 0, 0, 1}, {1, 0, 0, 1}
};

// Stepper parameters
const int STEPS_PER_REV = 4096;  // Correct for 28BYJ-48 half-step
const float DEG_PER_STEP = 360.0 / STEPS_PER_REV;
int current_steps = 2048;  // Start at 90° (2048 steps = 90°)

void setup() {
  pinMode(IN1, OUTPUT);
  pinMode(IN2, OUTPUT);
  pinMode(IN3, OUTPUT);
  pinMode(IN4, OUTPUT);
  Serial.begin(9600);
  Serial.println("Stepper ready!");
}

void stepMotor(int step) {
  int patternIndex = step % 8;
  if (patternIndex < 0) patternIndex += 8;  // Handle negative steps
  for (int i = 0; i < 4; i++) {
    digitalWrite(IN1, stepPattern[patternIndex][0]);
    digitalWrite(IN2, stepPattern[patternIndex][1]);
    digitalWrite(IN3, stepPattern[patternIndex][2]);
    digitalWrite(IN4, stepPattern[patternIndex][3]);
  }
  delayMicroseconds(2000);  // Tune for smoothness (1000-3000)
}

void moveToAngle(int target_angle) {
  // Limit angle to 30-150°
  if (target_angle < 30 || target_angle > 150) return;
  
  int target_steps = (int)(target_angle / DEG_PER_STEP);
  int steps_to_move = target_steps - current_steps;
  int direction = steps_to_move > 0 ? 1 : -1;
  steps_to_move = abs(steps_to_move);

  for (int i = 0; i < steps_to_move; i++) {
    stepMotor(current_steps + (i * direction));
  }
  current_steps = target_steps;
}

void loop() {
  if (Serial.available() > 0) {
    String data = Serial.readStringUntil('\n');
    int angle = data.toInt();
    moveToAngle(angle);
    Serial.print("Moved to ");
    Serial.print(angle);
    Serial.println(" degrees");
  }
}