#include <Arduino.h>

void setup()
{
    pinMode(2, OUTPUT);
    pinMode(3, OUTPUT);
    pinMode(4, OUTPUT);
    pinMode(5, OUTPUT);
    pinMode(6, OUTPUT);
    pinMode(7, OUTPUT);
    pinMode(8, OUTPUT);
    pinMode(9, OUTPUT);
    pinMode(10, OUTPUT);
    pinMode(11, OUTPUT);
    pinMode(12, OUTPUT);
    pinMode(13, OUTPUT);
}

void experiment(int in0, int in1, int in2, int in3, int in4, int in5, int in6, int in7, int in8, int b1, int b2, int b3)
{
    analogWrite(2, in0);
    analogWrite(3, in1);
    analogWrite(4, in2);
    analogWrite(5, in3);
    analogWrite(6, in4);
    analogWrite(7, in5);
    analogWrite(8, in6);
    analogWrite(9, in7);
    analogWrite(10, in8);
    analogWrite(11, b1);
    analogWrite(12, b2);
    analogWrite(13, b3);
}

void experiment1() { experiment(12, 17, 2, 154, 27, 112, 0, 46, 44, 60, 104, 89); }

void experiment2() { experiment(0, 75, 133, 16, 0, 1, 38, 73, 2, 3, 99, 93); }

void experiment3() { experiment(0, 4, 1, 104, 122, 12, 2, 44, 2, 1, 93, 100); }

void experiment4() { experiment(0, 91, 11, 31, 2, 2, 25, 97, 21, 4, 99, 92); }

void experiment5() { experiment(0, 8, 1, 8, 99, 10, 1, 28, 8, 32, 92, 97); }

void experiment6() { experiment(0, 153, 7, 1, 84, 19, 0, 49, 12, 82, 106, 86); }

void experiment7() { experiment(46, 26, 25, 34, 0, 29, 127, 40, 34, 43, 96, 98); }

void experiment8() { experiment(36, 129, 0, 8, 87, 89, 1, 34, 113, 82, 106, 86); }

void experiment9() { experiment(90, 79, 11, 5, 54, 137, 0, 42, 23, 36, 107, 89); }

void experiment10() { experiment(9, 187, 4, 39, 38, 7, 0, 6, 67, 74, 98, 105); }

void loop()
{
    // Specify which experiment to run
    experiment1();
}