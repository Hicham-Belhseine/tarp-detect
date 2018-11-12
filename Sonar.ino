/*********************************************************************
 *  FILE NAME: sonar-testing                                  
 *                                                            
 *  PURPOSE: This program shall record sonar data to file for 
 *           calculation of sonar error.                      
 *           
 *           
 * FILE REFERENCES:
 *
 * Name                   IO          Description
 * ------------           --          ---------------------------------
 * none
 * 
 * 
 * ASSUMPTIONS, CONSTRAINTS, RESTRICTIONS: None
 * 
 * 
 * NOTES:
 * This program is used with the HC-SR04 ultrasonic sensor and the 
 * Micro SD Card SDHC Mini TF Card Adapter
 * 
 * 
 * DEVELOPMENT HISTORY:
 * Date     Name             Release Description
 * -------- ---------------- ------- ---------------------------------
 * 10-12-18 Hicham Belhseine    1    Sonar and SD Card Code Added
 * 
 *********************************************************************/

#include <SD.h>
#include <SPI.h>

#define TRIGGER_PIN 6  /* trigger pin for ultrasonic sensor */
#define ECHO_PIN 5     /* echo pin for ultrasonic sensor    */
#define CM .034        /* centimeter conversion factor      */

#define CLOCK_PIN 10   /* SD Card Module Clock Pin          */


File   dataFile;       /* Data file for test data           */
double distance;       /* Object Distance [inches]          */

void setup( void ) {
    /* INITIALIZATION FOR ULTRASONIC SENSOR   */
    /* Set up pins for the ultrasonic sensors */
    pinMode(TRIGGER_PIN, OUTPUT);
    pinMode(ECHO_PIN, INPUT);

    /* INITIALIZATIN FOR SD CARD          */
    /* Set up pins for the SD Card Module */
    pinMode(CLOCK_PIN, OUTPUT);
    OpenFile();
    
    // Sets data rate in bits per second
    Serial.begin(9600);
}

void loop( void ) 
{
    distance = GetSonarReading();

    RecordSonarReading(distance);
    if(millis() > 15000)
    {
        CloseFile();
        Serial.print("Done Recording Data.");
        while(1){}
    }
}


/*********************************************************************
 * FUNCTION NAME: GetSonarReading
 * 
 * ARGUMENT LIST:
 * Argument        Type     IO  Description
 * --------------  -------  --  --------------------------------------
 * None
 * 
 * RETURN VALUE: Returns distance in cm between the ultrasonic sensor  
 *               and the object in front of it.
 *               
 *********************************************************************/

float GetSonarReading( void )
{
    long duration;    /* Time to detect emitted wave signal [ms] */
    double distance;  /* Object distance [cm]                    */
    
    /* Ensure that the trigger pin voltage is low */
    digitalWrite(TRIGGER_PIN, LOW);
    delayMicroseconds(2);
    
    /* Sets the trigger pin voltage to high for 10 us         */
    /* Most of the ultrasonic sensors have a trigger of 10 us */
    digitalWrite(TRIGGER_PIN, HIGH);
    delayMicroseconds(10);
    digitalWrite(TRIGGER_PIN, LOW);

    /* Time the sound wave took to go from the sensor */
    /* to the object and back                         */
    duration = pulseIn(ECHO_PIN, HIGH);
  
    /* Calculate the distance the wave traveled to the */
    /* object.                                         */
    distance = duration * CM / 2;
  
    return distance;

}

/*********************************************************************
 * FUNCTION NAME: OpenFile
 * 
 * ARGUMENT LIST:
 * Argument        Type     IO  Description
 * --------------  -------  --  --------------------------------------
 * None
 * 
 * RETURN VALUE: Opens "Sonar_Data.txt" file for data recording.
 *               
 *********************************************************************/

void OpenFile( void )
{
    dataFile = SD.open("Sonar_Data.txt", FILE_WRITE);
    
    return;
}

/*********************************************************************
 * FUNCTION NAME: CloseFile
 * 
 * ARGUMENT LIST:
 * Argument        Type     IO  Description
 * --------------  -------  --  --------------------------------------
 * None
 * 
 * RETURN VALUE: Closes the data file.
 *               
 *********************************************************************/

void CloseFile( void )
{
    /* If the file is open, close it */
    if( dataFile )
    {
        dataFile.close();
    }

    return;
}

/*********************************************************************
 * FUNCTION NAME: RecordSonarReading
 * 
 * ARGUMENT LIST:
 * Argument        Type     IO  Description
 * --------------  -------  --  --------------------------------------
 * distance        double   O   Ultrasonic Sensor Distance Recording 
 *                              in Inches
 * 
 * RETURN VALUE: Returns distance in cm between the ultrasonic sensor  
 *               and the object in front of it.
 *               
 *********************************************************************/

void RecordSonarReading(double distance)
{
    dataFile.println(distance);

    return;
}
