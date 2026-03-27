#include "ADC_lib.h"
#include <Arduino.h>



typedef struct adc_handle_t {
  struct cfg{
    bool smpl_cmplt;
    adc_pin_t input_pin;
    adc_sample_rate_t smpl_frq;
    uint16_t smpl_size;
  }cfg;
}adc_handle_t;

static adc_handle_t adc_handle;

adc_handle_t* adc_get_handle(void){
  return &adc_handle;
}

int8_t adc_config(struct adc_handle_t *handle, adc_sample_rate_t smpl_frq, uint16_t smpl_size, adc_pin_t pin) {
  int8_t r = 0;
  if(handle != NULL) {
    handle -> cfg.smpl_frq = smpl_frq;
    handle -> cfg.input_pin = pin;
    handle -> cfg.smpl_size = smpl_size;
  } else r = -1;
  return r;
}

int8_t adc_setup(struct adc_handle_t *handle) {
  int8_t r = 0;
  if(handle != NULL) {
    ADCSRA &= ~(bit(ADPS0) | bit(ADPS1) | bit(ADPS2));
    uint8_t MASK_A = 0, MASK_B = 0, MASK_C = 0;
    switch (handle->cfg.smpl_frq) {
      case KHZ_307_7:
        MASK_A = bit(ADPS1);
        break;
      case KHZ_153_8:
        MASK_A = bit(ADPS0);
        MASK_B = bit(ADPS1);
        break;
      case KHZ_76_9:
        MASK_A = bit(ADPS2);
        break;
      case KHZ_38_5:
        MASK_A = bit(ADPS0);
        MASK_B = bit(ADPS2);
        break;
      case KHZ_19_2:
        MASK_A = bit(ADPS1);
        MASK_B = bit(ADPS2);
        break;
      case KHZ_9_6:
        MASK_A = bit(ADPS0);
        MASK_B = bit(ADPS1);
        MASK_C = bit(ADPS2);
        break;
    }
    ADCSRA |= MASK_A | MASK_B | MASK_C;
    uint8_t adc_pin = handle->cfg.input_pin;
    for(uint8_t i = 0; i <21; i++) {
      analogRead(adc_pin);
    }
    handle -> cfg.smpl_cmplt = 0;
  } else r = -1;
  return r;
}


int8_t adc_software_trigger(struct adc_handle_t *handle, int16_t *smpl_buffer) {
  int8_t r = 0;
  if(handle != NULL) {
    uint8_t adc_pin = handle -> cfg.input_pin;
    uint16_t smpl_size = handle -> cfg.smpl_size;
    for(uint16_t i = 0; i < smpl_size; i++) {
      smpl_buffer[i] = analogRead(adc_pin);
    }
    handle -> cfg.smpl_cmplt = 1;
  } else r = -1;
  return r;
}

int8_t adc_software_trigger_32(struct adc_handle_t *handle, int32_t *smpl_buffer) {
  int8_t r = 0;
  if(handle != NULL) {
    uint8_t adc_pin = handle -> cfg.input_pin;
    uint16_t smpl_size = handle -> cfg.smpl_size;
    for(uint16_t i = 0; i < smpl_size; i++) {
      smpl_buffer[i] = (int32_t)analogRead(adc_pin);
    }
    handle -> cfg.smpl_cmplt = 1;
  } else r = -1;
  return r;
}

int8_t adc_clear_status(struct adc_handle_t *handle) {
  int8_t r = 0;
  if(handle != NULL) {
    handle -> cfg.smpl_cmplt = 0;
  } else r = -1;
  return r;
}

int8_t adc_smpl_status(struct adc_handle_t *handle) {
  int8_t r = 0;
  if(handle != NULL) {
    r = handle -> cfg.smpl_cmplt;
  } else r = -1;
  return r;
}

