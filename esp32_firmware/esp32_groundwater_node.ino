/*
 * ==============================================================================
 * Project: Groundwater Fingerprint — IoT Intelligence Node
 * Target: ESP32-S3-DevKitC-1-N8R8 (8MB Flash + 8MB PSRAM)
 * 
 * Hardware Connections:
 *   - HC-SR04 TRIG  -> GPIO 5
 *   - HC-SR04 ECHO  -> GPIO 18 (via 1k / 2k resistor divider to 3.3V)
 *   - Relay IN      -> GPIO 4 (3.3V logic compatible, switches 12V pump)
 *   - HC-SR04 VCC   -> 5V pin on ESP32-S3
 *   - HC-SR04 GND   -> GND
 *   - Relay VCC     -> 5V pin on ESP32-S3
 *   - Relay GND     -> GND
 *   - 12V Pump      -> Relay COM/NO & 12V 2A external DC adapter
 * ==============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>

// ---------------- Wi-Fi & Server Configuration ----------------
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Replace with the IP address of your computer running the ML server
const char* SERVER_URL    = "http://192.168.1.100:5050/api/telemetry";

// ---------------- Pin Definitions ----------------
const int PIN_TRIG        = 5;   // HC-SR04 Trigger pin
const int PIN_ECHO        = 18;  // HC-SR04 Echo pin (through divider)
const int PIN_RELAY       = 4;   // 1-channel relay control pin

// ---------------- Well Container Physical Dimensions ----------------
const float TANK_HEIGHT_CM   = 35.0; // Height of tank container
const float SENSOR_OFFSET_CM = 3.0;  // Offset distance from top

// ---------------- State Variables ----------------
bool pumpState = false;
bool isRegisteredWindow = true; // Permitted schedule flag
unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL_MS = 1000; // Send telemetry every 1 sec

// ---------------- Median Filter for HC-SR04 ----------------
float getFilteredDistance() {
  const int SAMPLES = 5;
  float readings[SAMPLES];

  for (int i = 0; i < SAMPLES; i++) {
    // Clear trigger
    digitalWrite(PIN_TRIG, LOW);
    delayMicroseconds(2);

    // 10 microsecond trigger pulse
    digitalWrite(PIN_TRIG, HIGH);
    delayMicroseconds(10);
    digitalWrite(PIN_TRIG, LOW);

    // Read echo pulse duration (timeout 25ms ~ 4.25m)
    long duration = pulseIn(PIN_ECHO, HIGH, 25000);

    if (duration == 0) {
      readings[i] = 999.0; // Timeout / out of range
    } else {
      // Speed of sound: 343 m/s = 0.0343 cm/us -> distance = (duration * 0.0343) / 2
      readings[i] = (duration * 0.0343) / 2.0;
    }
    delay(10);
  }

  // Simple bubble sort to find median
  for (int i = 0; i < SAMPLES - 1; i++) {
    for (int j = i + 1; j < SAMPLES; j++) {
      if (readings[i] > readings[j]) {
        float temp = readings[i];
        readings[i] = readings[j];
        readings[j] = temp;
      }
    }
  }

  // Return middle sample
  float medianDistance = readings[SAMPLES / 2];
  if (medianDistance > 400.0 || medianDistance < 2.0) {
    return -1.0; // Invalid reading
  }
  return medianDistance;
}

// ---------------- Pump Control ----------------
void setPump(bool state) {
  pumpState = state;
  // Most 5V relay modules are ACTIVE-LOW (LOW = ON, HIGH = OFF).
  // Adjust logic if your module is active-high.
  digitalWrite(PIN_RELAY, state ? LOW : HIGH);
  Serial.printf("[PUMP] Set to: %s\n", state ? "ON" : "OFF");
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n--- Groundwater Fingerprint IoT Node Initializing ---");

  // Configure Pins
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_RELAY, OUTPUT);
  setPump(false); // Default pump OFF

  // Connect to Wi-Fi
  Serial.printf("Connecting to Wi-Fi '%s'...", WIFI_SSID);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  
  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\nConnected! IP Address: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\nWi-Fi connection pending. Continuing in autonomous sensing mode.");
  }
}

void loop() {
  unsigned long now = millis();

  // Handle Serial Commands for Testing ('1'=Pump ON, '0'=Pump OFF, 's'=Simulate cycle)
  if (Serial.available()) {
    char cmd = Serial.read();
    if (cmd == '1') setPump(true);
    else if (cmd == '0') setPump(false);
    else if (cmd == 'r') isRegisteredWindow = !isRegisteredWindow;
  }

  // Periodic Telemetry Transmission
  if (now - lastTelemetryTime >= TELEMETRY_INTERVAL_MS) {
    lastTelemetryTime = now;

    float distance = getFilteredDistance();
    if (distance > 0) {
      float waterLevel = (TANK_HEIGHT_CM + SENSOR_OFFSET_CM) - distance;
      waterLevel = constrain(waterLevel, 0.0, TANK_HEIGHT_CM);

      Serial.printf("[TELEMETRY] Dist: %.2f cm | Level: %.2f cm | Pump: %d | Registered: %d\n",
                    distance, waterLevel, pumpState ? 1 : 0, isRegisteredWindow ? 1 : 0);

      // Send to server if connected
      if (WiFi.status() == WL_CONNECTED) {
        HTTPClient http;
        http.begin(SERVER_URL);
        http.addHeader("Content-Type", "application/json");

        String payload = "{";
        payload += "\"node_id\":\"ESP32S3-GW-01\",";
        payload += "\"distance_cm\":" + String(distance, 2) + ",";
        payload += "\"water_level_cm\":" + String(waterLevel, 2) + ",";
        payload += "\"pump_state\":" + String(pumpState ? 1 : 0) + ",";
        payload += "\"is_registered\":" + String(isRegisteredWindow ? 1 : 0);
        payload += "}";

        int httpResponseCode = http.POST(payload);
        if (httpResponseCode > 0) {
          String response = http.getString();
          // Check if server commanded pump action
          if (response.indexOf("\"command\":\"PUMP_ON\"") >= 0) setPump(true);
          else if (response.indexOf("\"command\":\"PUMP_OFF\"") >= 0) setPump(false);
        }
        http.end();
      }
    }
  }
}
