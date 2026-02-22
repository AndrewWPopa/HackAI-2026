#include <LiquidCrystal.h>
#include <math.h>
#include <avr/pgmspace.h>

//Pin assignment
const int rs = 4, en = 6, d4 = 10, d5 = 11, d6 = 12, d7 = 13;
LiquidCrystal lcd(rs, en, d4, d5, d6, d7);

uint8_t input_pixels[INPUT_SIZE];

float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

void softmax(float* z, float* out, int n) {
  float m = z[0]; for(int i=1;i<n;i++) if(z[i]>m) m=z[i];
  float s = 0.0f; for(int i=0;i<n;i++) { out[i]=expf(z[i]-m); s+=out[i]; }
  for(int i=0;i<n;i++) out[i]/=s;
}

int predict() {
  float h1[H1_SIZE] = {0};
  float h2[H2_SIZE] = {0};
  float logits[OUTPUT_SIZE] = {0};
  float probs[OUTPUT_SIZE] = {0};

  // Layer 1
  for(int i = 0; i < H1_SIZE; i++) {
    float sum = pgm_read_float(&bias0[i]);
    for(int j = 0; j < INPUT_SIZE; j++) {
      sum += (input_pixels[j] / 255.0f) * pgm_read_float(&weights0[j][i]);
    }
    h1[i] = sigmoid(sum);
  }

  // Layer 2
  for(int i = 0; i < H2_SIZE; i++) {
    float sum = pgm_read_float(&bias1[i]);
    for(int j = 0; j < H1_SIZE; j++) {
      sum += h1[j] * pgm_read_float(&weights1[j][i]);
    }
    h2[i] = sigmoid(sum);
  }

  // Output
  for(int i = 0; i < OUTPUT_SIZE; i++) {
    float sum = pgm_read_float(&bias2[i]);
    for(int j = 0; j < H2_SIZE; j++) {
      sum += h2[j] * pgm_read_float(&weights2[j][i]);
    }
    logits[i] = sum;
  }

  softmax(logits, probs, OUTPUT_SIZE);

  int pred = 0;
  float mx = probs[0];
  for(int i = 1; i < OUTPUT_SIZE; i++) {
    if(probs[i] > mx) {
      mx = probs[i];
      pred = i;
    }
  }
  return pred;
}

// crc function for debugging
uint16_t crc16(const uint8_t* data, uint16_t len) {
  uint16_t crc = 0xFFFF;
  for (uint16_t i = 0; i < len; i++) {
    crc ^= data[i];
    for (uint8_t j = 0; j < 8; j++) {
      if (crc & 1) crc = (crc >> 1) ^ 0x8408;
      else crc >>= 1;
    }
  }
  return crc;
}

void setup() {
  Serial.begin(115200);
  lcd.begin(16, 2);
  lcd.print("MNIST NN Ready");
  lcd.setCursor(0,1);
  // Print random for debugging
  // randomSeed(analogRead(A0));
  // lcd.print(random(1000));
  Serial.println("Ready - Send 784 pixels (0-255)");
  // Clear extra bytes in buffer
  while (Serial.available() > 0) Serial.read();
}

void loop() {
  static uint16_t received = 0;

  // Accumulate bytes safely
  while (Serial.available() > 0 && received < INPUT_SIZE) {
    input_pixels[received++] = Serial.read();
  }

  // When there are exactly 784 pixels then process
  if (received == INPUT_SIZE) {
    lcd.clear();
    lcd.print("Processing...");

    int digit = predict();

    lcd.clear();
    lcd.print("Predicted:");
    lcd.setCursor(0, 1);
    lcd.print(digit);
    
    // Lines for debugging to print crc of whole matrix, and then certain pixels in the image
    // lcd.print("-");
    // lcd.print(crc16(input_pixels, INPUT_SIZE));
    // lcd.print("-");
    // lcd.print(input_pixels[178]);
    // lcd.print("-");
    // lcd.print(input_pixels[179]);

    Serial.print("PREDICTION:");
    Serial.println(digit);

    while (Serial.available() > 0) Serial.read();
    received = 0;
  }

}