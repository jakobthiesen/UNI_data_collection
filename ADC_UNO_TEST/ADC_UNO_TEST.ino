#include <Arduino.h>
#include "ADC_lib.h"

#define SMPL_SIZE 512
#define SAMPLE_RATE 10E3
#define START_CMD 'r'
#define END_CHAR '\n'


volatile uint32_t isr_count =0;
volatile bool isr_active = false;
volatile bool capture_done = false;


int16_t smpl_buffer[SMPL_SIZE];
adc_handle_t* adc_handle = adc_get_handle();

uint32_t timer = 0;

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


static inline uint16_t adc_read_fixed_channel() {
  ADCSRA |= (1 << ADSC);
  while(ADCSRA & (1 << ADSC)) {

  }
  return ADC;
}

void start_capture() {
  noInterrupts();
  isr_count = 0;
  capture_done = false;
  isr_active = true;
  interrupts();
}

void tx_samples() {
  uint16_t count = SMPL_SIZE;
  Serial.write((uint8_t*)&count, sizeof(count));

  Serial.write((uint8_t*)smpl_buffer, sizeof(smpl_buffer));

  Serial.write(END_CHAR);
}


ISR(TIMER1_COMPA_vect) {
  if (!isr_active) return;

  if (isr_count < SMPL_SIZE) {
    smpl_buffer[isr_count] = adc_read_fixed_channel();
    isr_count++;
  }

  if (isr_count >= SMPL_SIZE) {
    isr_active = false;
    capture_done = true;
  }
}




void setup() {
  // put your setup code here, to run once:
  Serial.begin(115200);
  adc_config(adc_handle, KHZ_153_8, SMPL_SIZE, ADC_A0);
  adc_setup(adc_handle);

  delay(100);
  setup_adc_isr_timer(50000);
  // isr_active = 1;
}


void loop() {
  // if(isr_active == 0) {
  //   for(uint16_t n = 0; n < SMPL_SIZE; n++) {
  //     Serial.println(smpl_buffer[n]);
  //     delay(3);
  //   }
  //   delay(500);
  //   isr_count = 0;
  //   isr_active = 1;
  // }
  if (Serial.available()) {
    char c = Serial.read();
    if (c == START_CMD && !isr_active) {
      start_capture();
    }
  }
  if(capture_done) {
    noInterrupts();
    capture_done = false;
    interrupts();

    tx_samples();
  }
}




