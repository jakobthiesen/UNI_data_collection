#include "pins_arduino.h"
#ifndef ADC_LIB_H
#define ADC_LIB_H

#include <Arduino.h>

#ifdef __cplusplus
extern "C" {
#endif


typedef enum {
  KHZ_307_7,
  KHZ_153_8,
  KHZ_76_9,
  KHZ_38_5,
  KHZ_19_2,
  KHZ_9_6
} adc_sample_rate_t;


typedef uint8_t adc_pin_t;

#define ADC_A0 ((adc_pin_t)A0)
#define ADC_A1 ((adc_pin_t)A1)
#define ADC_A2 ((adc_pin_t)A2)
#define ADC_A3 ((adc_pin_t)A3)
#define ADC_A4 ((adc_pin_t)A4)
#define ADC_A5 ((adc_pin_t)A5)

typedef struct adc_handle_t adc_handle_t;


adc_handle_t* adc_get_handle(void);
int8_t adc_software_trigger(struct adc_handle_t *handle, int16_t *smpl_buffer);
int8_t adc_software_trigger_32(struct adc_handle_t *handle, int32_t *smpl_buffer);
int8_t adc_smpl_status(struct adc_handle_t *handle);
int8_t adc_config(struct adc_handle_t *handle, adc_sample_rate_t smpl_frq, uint16_t smpl_size, adc_pin_t pin);
int8_t adc_setup(struct adc_handle_t *handle);
int8_t adc_clear_status(struct adc_handle_t *handle);


#ifdef __cplusplus
}
#endif

#endif

