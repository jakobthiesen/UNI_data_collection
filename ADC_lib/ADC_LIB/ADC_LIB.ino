#include <Arduino.h>
#include "ADC_lib.h"

#define SMPL_SIZE 512

int16_t smpl_buffer[SMPL_SIZE];
adc_handle_t* adc_handle = adc_get_handle();

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  adc_config(adc_handle, KHZ_153_8, SMPL_SIZE, ADC_A0);
  adc_setup(adc_handle);
  delay(100);
}

void loop() {
  adc_software_trigger(adc_handle, smpl_buffer);
  for(uint16_t i = 0; i < SMPL_SIZE; i ++) {
    Serial.println(smpl_buffer[i]);
  }
  delay(500);
}


