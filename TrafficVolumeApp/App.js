import React, { useState } from 'react';
import { View, TextInput, Button, Text, StyleSheet, ScrollView, Alert } from 'react-native';

export default function App() {
  const [temperature, setTemperature] = useState('');
  const [dayOfWeek, setDayOfWeek] = useState('');
  const [location, setLocation] = useState('');
  const [timeOfDay, setTimeOfDay] = useState('');
  const [prediction, setPrediction] = useState(null);

  const predictTraffic = async () => {
    if (!temperature || !dayOfWeek || !location || !timeOfDay) {
      Alert.alert('All fields are required');
      return;
    }

    try {
      const response = await fetch('https://your-flask-app.onrender.com/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          temperature: parseFloat(temperature),
          day_of_week: dayOfWeek,
          location: location,
          time_of_day: timeOfDay,
        }),
      });

      const data = await response.json();
      setPrediction(data.traffic_volume);
    } catch (error) {
      console.error(error);
      Alert.alert('Error connecting to API');
    }
  };

  return (
    <ScrollView contentContainerStyle={styles.container}>
      <Text style={styles.heading}>Traffic Volume Predictor</Text>

      <TextInput
        style={styles.input}
        placeholder="Temperature"
        keyboardType="numeric"
        value={temperature}
        onChangeText={setTemperature}
      />
      <TextInput
        style={styles.input}
        placeholder="Day of the Week"
        value={dayOfWeek}
        onChangeText={setDayOfWeek}
      />
      <TextInput
        style={styles.input}
        placeholder="Location"
        value={location}
        onChangeText={setLocation}
      />
      <TextInput
        style={styles.input}
        placeholder="Time of Day"
        value={timeOfDay}
        onChangeText={setTimeOfDay}
      />

      <Button title="Predict" onPress={predictTraffic} />

      {prediction && (
        <Text style={styles.result}>
          Predicted Traffic Volume: {prediction}
        </Text>
      )}
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    padding: 20,
    paddingTop: 60,
    backgroundColor: '#fff',
    flexGrow: 1,
  },
  heading: {
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 30,
    textAlign: 'center',
  },
  input: {
    borderWidth: 1,
    borderColor: '#aaa',
    marginBottom: 15,
    padding: 10,
    borderRadius: 5,
  },
  result: {
    marginTop: 20,
    fontSize: 18,
    fontWeight: 'bold',
    color: 'green',
    textAlign: 'center',
  },
});
