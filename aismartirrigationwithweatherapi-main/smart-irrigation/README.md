# Smart Irrigation System Documentation

## Overview
The Smart Irrigation System automates the process of watering plants based on weather conditions, soil moisture, and other environmental factors. This project ensures optimal water usage, promotes plant health, and contributes to overall sustainability.

## Architecture
The system consists of the following components:
- **Sensors**: Measure soil moisture, temperature, and weather data.
- **Controller**: Receives data from sensors and decides when to water the plants.
- **Actuators**: Control the watering system.
- **User Interface**: Allows users to configure settings and view system status.

## Requirements
- Arduino or Raspberry Pi
- Moisture sensors
- Temperature sensors
- Wi-Fi module
- Relay module for controlling the water supply
- Node.js for the backend

## Quick Start Guide
1. Clone the repository: `git clone https://github.com/mummadimaheswar/aismartirrigationwithweatherapi.git`
2. Install required libraries and dependencies.
3. Connect the sensors and actuators to your microcontroller.
4. Configure the system parameters in `config.js`.
5. Run the application: `node app.js`

## Configuration
Edit the `config.js` file to set up:
- Pin numbers for sensors and actuators
- Wi-Fi credentials
- Thresholds for moisture levels

## Example Output
The system provides real-time data output showing:
- Current soil moisture levels
- Weather conditions
- Watering activity and history

## Project Structure
```
├── README.md
├── config.js
├── app.js
├── sensors
│   ├── moistureSensor.js
│   └── temperatureSensor.js
├── actuators
│   └── relayControl.js
└── data
    ├── logs.json
    └── configurations.json
```

This structure separates the functionality clearly, allowing for easy maintenance and scalability.