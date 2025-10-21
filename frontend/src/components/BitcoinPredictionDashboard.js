import React, { useState, useEffect } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Container, 
  Form, 
  Button, 
  Spinner 
} from 'react-bootstrap';
import Plot from 'react-plotly.js';

const BitcoinPredictionDashboard = () => {
  const [selectedModel, setSelectedModel] = useState('naive');
  const [windowSize, setWindowSize] = useState(7);
  const [horizon, setHorizon] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);

  // Model options for the dropdown
  const modelOptions = [
    { value: 'naive', label: 'Naive Forecast' },
    { value: 'dense71', label: 'Dense Model (Window=7, Horizon=1)' },
    { value: 'dense301', label: 'Dense Model (Window=30, Horizon=1)' },
    { value: 'dense307', label: 'Dense Model (Window=30, Horizon=7)' },
    { value: 'conv1d', label: 'Conv1D Model' },
    { value: 'lstm', label: 'LSTM Model' },
    { value: 'multivariate', label: 'Multivariate Dense Model' },
    { value: 'nbeats', label: 'N-BEATS Algorithm' },
    { value: 'ensemble', label: 'Ensemble Model' },
    { value: 'future', label: 'Future Prediction Model' }
  ];

  const handlePredict = async () => {
    setIsLoading(true);
    
    // Simulate API call to backend
    // In a real implementation, this would call your backend API
    setTimeout(() => {
      // Mock prediction data
      const mockData = {
        historical: {
          x: Array.from({length: 30}, (_, i) => `2023-01-${(i+1).toString().padStart(2, '0')}`),
          y: Array.from({length: 30}, (_, i) => 30000 + Math.random() * 5000 - 2500),
          type: 'scatter',
          mode: 'lines',
          name: 'Historical Price'
        },
        predictions: {
          x: Array.from({length: horizon}, (_, i) => `2023-01-${(31+i).toString().padStart(2, '0')}`),
          y: Array.from({length: horizon}, () => 32000 + Math.random() * 3000 - 1500),
          type: 'scatter',
          mode: 'lines+markers',
          name: 'Predictions'
        }
      };
      
      setPredictionData(mockData);
      setIsLoading(false);
    }, 1500);
  };

  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card className="mb-4">
            <Card.Header>
              <h3>Bitcoin Price Prediction Dashboard</h3>
            </Card.Header>
            <Card.Body>
              <Form>
                <Row>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Select Model</Form.Label>
                      <Form.Select 
                        value={selectedModel} 
                        onChange={(e) => setSelectedModel(e.target.value)}
                      >
                        {modelOptions.map((option) => (
                          <option key={option.value} value={option.value}>
                            {option.label}
                          </option>
                        ))}
                      </Form.Select>
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Window Size</Form.Label>
                      <Form.Control 
                        type="number" 
                        value={windowSize} 
                        onChange={(e) => setWindowSize(parseInt(e.target.value))}
                        min="1"
                        max="100"
                      />
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group className="mb-3">
                      <Form.Label>Horizon</Form.Label>
                      <Form.Control 
                        type="number" 
                        value={horizon} 
                        onChange={(e) => setHorizon(parseInt(e.target.value))}
                        min="1"
                        max="30"
                      />
                    </Form.Group>
                  </Col>
                </Row>
                <Button 
                  variant="primary" 
                  onClick={handlePredict}
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Spinner
                        as="span"
                        animation="border"
                        size="sm"
                        role="status"
                      />
                      {' '}Predicting...
                    </>
                  ) : 'Generate Prediction'}
                </Button>
              </Form>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {predictionData && (
        <Row>
          <Col md={12}>
            <Card>
              <Card.Header>
                <h4>Prediction Results</h4>
              </Card.Header>
              <Card.Body>
                <Plot
                  data={[
                    predictionData.historical,
                    predictionData.predictions
                  ]}
                  layout={{
                    title: 'Bitcoin Price Prediction',
                    xaxis: { title: 'Date' },
                    yaxis: { title: 'Price (USD)' },
                    showlegend: true,
                    width: '100%',
                    height: 500
                  }}
                  config={{ responsive: true }}
                />
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}
    </Container>
  );
};

export default BitcoinPredictionDashboard;