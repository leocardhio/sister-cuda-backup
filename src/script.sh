gcc -g -Wall -o serial serial.c -lm

if [ -e serial ]; then
    ./serial ./tc/K04-04-TC1 ../result/K04-04-TC1_Serial.txt
    ./serial ./tc/K04-04-TC2 ../result/K04-04-TC2_Serial.txt
    ./serial ./tc/K04-04-TC3 ../result/K04-04-TC3_Serial.txt
    ./serial ./tc/K04-04-TC4 ../result/K04-04-TC4_Serial.txt
fi