
#include <Arduino.h>
#include "LTC1451_lib.h"
#include "Apollo3_ADC_LIB.h"


typedef enum RGB_COLOUR {
  LED_R,
  LED_G,
  LED_B,
  LED_OFF
}RGB_COLOUR_t;

#define SMPL_SIZE 25
#define SAMPLE_RATE 10E3

#define R_LED 5
#define G_LED 7
#define B_LED 6

LTC1451_struct DAC;

uint8_t led_pins[3] = {R_LED, G_LED, B_LED};

volatile uint32_t isr_count =0;
volatile bool isr_active = false;

volatile bool smpl_ready = 0;
float mean_r = 0;
float mean_g = 0;
float mean_b = 0;

uint32_t timer = 0;

int32_t smpl_buffer[SMPL_SIZE];
adc_handle_t* adc_handle = adc_get_handle();

void set_led_colour(uint8_t* led_pins, RGB_COLOUR_t led_colour) {
  for(uint8_t n = 0; n <3; n++) {
    digitalWrite(led_pins[n], LOW);
  }
  if(led_colour != LED_OFF) {
    digitalWrite(led_pins[led_colour], HIGH);
  }
}

void set_led_current(float current) {
  float v_dac = (current*22.0+0.00571429)/0.04761905;
  v_dac *= 1.00864;
  DAC.setV(&DAC, v_dac);
}


float measure_rgb(uint8_t* led_pins, RGB_COLOUR_t led_colour) {
  float mean_off = 0;
  float mean_on = 0;
  set_led_colour(led_pins, LED_OFF);
  delay(2);
  adc_software_burst_trigger(adc_handle, smpl_buffer);
  while(1) {
    if(adc_smpl_status(adc_handle)) break;
  }
    adc_transfer_data(adc_handle, smpl_buffer);
    adc_arm_burst_scan(adc_handle, smpl_buffer);
    adc_clear_status(adc_handle);
  for(uint16_t n = 0; n<SMPL_SIZE; n++) {
    mean_off = mean_off + float(smpl_buffer[n]);
  }
  mean_off /= float(SMPL_SIZE);

  set_led_colour(led_pins, led_colour);
  delay(2);
  adc_software_burst_trigger(adc_handle, smpl_buffer);
  while(1) {
    if(adc_smpl_status(adc_handle)) break;
  }
    adc_transfer_data(adc_handle, smpl_buffer);
    adc_arm_burst_scan(adc_handle, smpl_buffer);
    adc_clear_status(adc_handle);
  for(uint16_t n = 0; n<SMPL_SIZE; n++) {
    mean_on = mean_on + float(smpl_buffer[n]);
  }
  mean_on /= float(SMPL_SIZE);
  mean_on = mean_on-mean_off;
  return(mean_on);
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  adc_config(adc_handle, 10E3, SMPL_SIZE, BURST_SCAN, ADC_A0, OSR_8, ADC_14BIT);
  adc_arm_burst_scan(adc_handle, smpl_buffer);
  adc_setup(adc_handle, smpl_buffer);
  //CLK, Din, nCS
  init_LTC1451(&DAC, 2, 3, 4);
  delay(100);
  // DAC.setBin(&DAC, 100);
  DAC.cfg.v_ref = 2.0497*2.0;
  
  // DAC.setV(&DAC, 0.24);
  set_led_current(0.009);
  // DAC.setBin(&DAC, 3000);
  pinMode(R_LED, OUTPUT);
  pinMode(G_LED, OUTPUT);
  pinMode(B_LED, OUTPUT);
  set_led_colour(led_pins, LED_OFF);

  // digitalWrite(R_LED, LOW);
  // digitalWrite(G_LED, LOW);
  // digitalWrite(B_LED, LOW);

  // Serial.println("Start");

}

void loop() {
  // put your main code here, to run repeatedly:

  mean_r = measure_rgb(led_pins, LED_R);
  mean_g = measure_rgb(led_pins, LED_G);
  mean_b = measure_rgb(led_pins, LED_B);

  Serial.print("R:");
  Serial.print(mean_r);
  Serial.print(",");
  Serial.print("g:");
  Serial.print(mean_g);
  Serial.print(",");
  Serial.print("B:");
  Serial.println(mean_b);


  // if(smpl_ready == 0) {
  //   adc_software_burst_trigger(adc_handle, smpl_buffer);
  //   smpl_ready = 1;
  // }
  // if(adc_smpl_status(adc_handle)) {
  //   smpl_ready = 0;
  //   adc_transfer_data(adc_handle, smpl_buffer);
  //   adc_arm_burst_scan(adc_handle, smpl_buffer);
  //   adc_clear_status(adc_handle);
  //   for(uint16_t n = 0; n<SMPL_SIZE; n++) {
  //     mean = mean + float(smpl_buffer[n]);
  //   }
  //   mean /= float(SMPL_SIZE);
  //   Serial.println(mean);
  //   mean = 0;
  //   delay(1);
  // }
}



