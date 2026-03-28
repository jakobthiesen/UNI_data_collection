
#include <Arduino.h>
#include "LTC1451_lib.h"
#include "Apollo3_ADC_LIB.h"

#define SMPL_SIZE 25
#define SAMPLE_RATE 10E3

#define R_LED 5
#define G_LED 7
#define B_LED 6

#define START_CMD 'r'
#define END_CHAR '\n'

typedef enum RGB_COLOUR {
  LED_R,
  LED_G,
  LED_B,
  LED_OFF
} RGB_COLOUR_t;

typedef struct rgb_handle_t {
  float ambient = 0;
  float r_val = 0;
  float g_val = 0;
  float b_val = 0;
} rgb_handle_t;

LTC1451_struct DAC;

uint8_t led_pins[3] = { R_LED, G_LED, B_LED };

volatile uint32_t isr_count = 0;
volatile bool isr_active = false;
volatile bool capture_done = false;


volatile bool smpl_ready = 0;
float mean_r = 0;
float mean_g = 0;
float mean_b = 0;

uint32_t timer = 0;

int32_t smpl_buffer[SMPL_SIZE];
adc_handle_t* adc_handle = adc_get_handle();
rgb_handle_t rgb_handle;

void tx_rgb_data(float amb_val, float r_val, float g_val, float b_val) {
  Serial.write((uint8_t*)&amb_val, sizeof(amb_val));
  Serial.write((uint8_t*)&r_val, sizeof(r_val));
  Serial.write((uint8_t*)&g_val, sizeof(g_val));
  Serial.write((uint8_t*)&b_val, sizeof(b_val));
  Serial.write(END_CHAR);
}

void set_led_colour(uint8_t* led_pins, RGB_COLOUR_t led_colour) {
  if (led_colour == LED_OFF) {
    for (uint8_t n = 0; n < 3; n++) {
      digitalWrite(led_pins[n], LOW);
    }
  } else {
    for (uint8_t n = 0; n < 3; n++) {
      digitalWrite(led_pins[n], HIGH);
    }
    for (uint8_t n = 0; n < 3; n++) {
      if (n != led_colour) {
        digitalWrite(led_pins[n], LOW);
      }
    }
  }
}

void set_led_current(float current) {
  float v_dac = (current * 22.0 + 0.00571429) / 0.04761905;
  v_dac *= 1.00864;
  DAC.setV(&DAC, v_dac);
}

float get_adc_data(void) {
  adc_software_burst_trigger(adc_handle, smpl_buffer);
  while (1) {
    if (adc_smpl_status(adc_handle)) break;
  }
  adc_transfer_data(adc_handle, smpl_buffer);
  adc_arm_burst_scan(adc_handle, smpl_buffer);
  adc_clear_status(adc_handle);
  float mean = 0;
  for (uint16_t n = 0; n < SMPL_SIZE; n++) {
    mean = mean + float(smpl_buffer[n]);
  }
  mean /= float(SMPL_SIZE);
  return (mean);
}



void measure_rgb(rgb_handle_t* handle, float current) {
  float mean_off = 0;
  float mean_on = 0;
  set_led_current(0);
  // set_led_colour(led_pins, LED_OFF);
  delay(10);
  mean_off = get_adc_data();
  handle->ambient = mean_off;

  set_led_current(current);
  set_led_colour(led_pins, LED_R);
  delay(10);

  mean_on = get_adc_data();
  handle->r_val = mean_on - mean_off;

  set_led_colour(led_pins, LED_G);
  delay(10);

  mean_on = get_adc_data();
  handle->g_val = mean_on - mean_off;

  set_led_colour(led_pins, LED_B);
  delay(10);

  mean_on = get_adc_data();
  handle->b_val = mean_on - mean_off;

  set_led_current(0);
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  adc_config(adc_handle, 20E3, SMPL_SIZE, BURST_SCAN, ADC_A0, OSR_16, ADC_14BIT);
  adc_arm_burst_scan(adc_handle, smpl_buffer);
  adc_setup(adc_handle, smpl_buffer);
  //CLK, Din, nCS
  init_LTC1451(&DAC, 2, 3, 4);
  delay(100);
  // DAC.setBin(&DAC, 100);
  DAC.cfg.v_ref = 2.0497 * 2.0;

  // DAC.setV(&DAC, 0.24);
  set_led_current(0.000);
  // DAC.setBin(&DAC, 3000);
  pinMode(R_LED, OUTPUT);
  pinMode(G_LED, OUTPUT);
  pinMode(B_LED, OUTPUT);
  // set_led_colour(led_pins, LED_OFF);

  // digitalWrite(R_LED, LOW);
  // digitalWrite(G_LED, LOW);
  // digitalWrite(B_LED, LOW);

  // Serial.println("Start");
}

void loop() {
  // put your main code here, to run repeatedly:

  if (Serial.available()) {
    String cmd = Serial.readStringUntil('\n');

    if (cmd[0] == START_CMD) {
      int int_current = cmd.substring(1).toInt();             // "r128" → 128
      float current = (float(int_current) / 255.0f) * 0.01f;  // 0–10 mA
      // Serial.println(current*1000.0);
      delay(1);
      measure_rgb(&rgb_handle, current);
      capture_done = 1;
      delay(1);
    }
  }

  if (capture_done) {
    tx_rgb_data(rgb_handle.ambient, rgb_handle.r_val, rgb_handle.g_val, rgb_handle.b_val);
    capture_done = 0;
  }


  // measure_rgb(&rgb_handle, 0.008);
  // Serial.print("Amb:");
  // Serial.print(rgb_handle.ambient);
  // Serial.print(",");
  // Serial.print("R:");
  // Serial.print(rgb_handle.r_val);
  // Serial.print(",");
  // Serial.print("g:");
  // Serial.print(rgb_handle.g_val);
  // Serial.print(",");
  // Serial.print("B:");
  // Serial.println(rgb_handle.b_val);



  // float current = 0.00004;
  // mean_r = measure_rgb(led_pins, LED_R, current);
  // mean_g = measure_rgb(led_pins, LED_G, current);
  // mean_b = measure_rgb(led_pins, LED_B, current);
  // Serial.print("R:");
  // Serial.print(mean_r);
  // Serial.print(",");
  // Serial.print("g:");
  // Serial.print(mean_g);
  // Serial.print(",");
  // Serial.print("B:");
  // Serial.println(mean_b);


  // set_led_colour(led_pins, LED_G);
  // set_led_current(0.0095);
  // delay(10);
  // if(smpl_ready == 0) {
  //   adc_software_burst_trigger(adc_handle, smpl_buffer);
  //   smpl_ready = 1;
  // }
  // if(adc_smpl_status(adc_handle)) {
  //   smpl_ready = 0;
  //   float mean = 0;
  //   adc_transfer_data(adc_handle, smpl_buffer);
  //   adc_arm_burst_scan(adc_handle, smpl_buffer);
  //   adc_clear_status(adc_handle);
  //   for(uint16_t n = 0; n<SMPL_SIZE; n++) {
  //     mean = mean + float(smpl_buffer[n]);
  //     // Serial.println(smpl_buffer[n]);
  //     // delay(2);
  //   }
  //   mean /= float(SMPL_SIZE);
  //   Serial.println(mean);
  //   delay(1);
  // }
}
