
#include <Arduino.h>
#include "LTC1451_lib.h"
#include "ADC_lib.h"


typedef enum RGB_COLOUR {
  LED_R,
  LED_G,
  LED_B,
  LED_OFF
}RGB_COLOUR_t;

#define SMPL_SIZE 512
#define SAMPLE_RATE 10E3

#define R_LED 5
#define G_LED 7
#define B_LED 6

LTC1451_struct DAC;

uint8_t led_pins[3] = {R_LED, G_LED, B_LED};

volatile uint32_t isr_count =0;
volatile bool isr_active = false;

int16_t smpl_buffer[SMPL_SIZE];
adc_handle_t* adc_handle = adc_get_handle();

void set_led_colour(uint8_t* led_pins, RGB_COLOUR_t led_color) {
  for(uint8_t n = 0; n <3; n++) {
    digitalWrite(led_pins[n], LOW);
  }
  if(led_color != LED_OFF) {
    digitalWrite(led_pins[led_color], HIGH);
  }
}

void set_led_current(float current) {
  float v_dac = (current*22.0+0.00571429)/0.04761905;
  v_dac *= 1.00864;
  DAC.setV(&DAC, v_dac);
}


void setup_adc_isr_timer(uint16_t frq) {
  noInterrupts();
  TCCR1A = 0;
  TCCR1B = 0;
  TCNT1 = 0;

  float m_clk = 16E6;
  float prescaler = 8;
  m_clk = m_clk/prescaler;
  float target = float(frq);
  target = (m_clk/target)-1;
  OCR1A = uint16_t(target);

  TCCR1B |= (1 << WGM12);
  TCCR1B |= (1 << CS11);

  TIMSK1 |= (1 << OCIE1A);

  interrupts();
}

ISR(TIMER1_COMPA_vect) {
  if(isr_active == 1) {
    if (isr_count < SMPL_SIZE) {
      smpl_buffer[isr_count] = adc_read_fixed_channel();
      isr_count++;
    }

    if (isr_count >= SMPL_SIZE) {
      isr_active = false;
    }
  }
}

static inline uint16_t adc_read_fixed_channel() {
  ADCSRA |= (1 << ADSC);
  while(ADCSRA & (1 << ADSC)) {

  }
  return ADC;
}

void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  adc_config(adc_handle, KHZ_153_8, SMPL_SIZE, ADC_A0);
  adc_setup(adc_handle);
  setup_adc_isr_timer(1000);
  
  //CLK, Din, nCS
  init_LTC1451(&DAC, 2, 3, 4);
  delay(100);
  // DAC.setBin(&DAC, 100);
  DAC.cfg.v_ref = 2.0497*2.0;
  
  // DAC.setV(&DAC, 0.24);
  set_led_current(0.005);
  // DAC.setBin(&DAC, 3000);
  pinMode(R_LED, OUTPUT);
  pinMode(G_LED, OUTPUT);
  pinMode(B_LED, OUTPUT);
  set_led_colour(led_pins, LED_B);

  // digitalWrite(R_LED, LOW);
  // digitalWrite(G_LED, LOW);
  // digitalWrite(B_LED, LOW);
  isr_active = 1;

}

void loop() {
  // put your main code here, to run repeatedly:

  if(isr_active == 0) {
    for(uint16_t n = 0; n < SMPL_SIZE; n++) {
      Serial.println(smpl_buffer[n]);
      delay(5);
    }
    delay(500);
    isr_count = 0;
    isr_active = 1;
  }
}



