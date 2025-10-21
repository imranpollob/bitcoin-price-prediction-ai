import React, { useState, useEffect } from 'react';
import { 
  Row, 
  Col, 
  Card, 
  Container, 
  Form, 
  Button, 
  Spinner,
  Alert
} from 'react-bootstrap';
import Plot from 'react-plotly.js';

const BitcoinPredictionDashboard = ({ modelType }) => {
  const [selectedModel, setSelectedModel] = useState(modelType || 'naive');
  const [windowSize, setWindowSize] = useState(7);
  const [horizon, setHorizon] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [predictionData, setPredictionData] = useState(null);
  const [metrics, setMetrics] = useState(null);

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

  // Update selected model when modelType prop changes
  useEffect(() => {
    if (modelType) {
      setSelectedModel(modelType);
    }
  }, [modelType]);

  const handlePredict = async () => {
    setIsLoading(true);
    
    try {
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
        
        // Mock metrics
        const mockMetrics = {
          mae: (Math.random() * 1000 + 500).toFixed(2),
          rmse: (Math.random() * 1500 + 700).toFixed(2),
          mape: (Math.random() * 5 + 2).toFixed(2),
          mase: (Math.random() * 2 + 0.5).toFixed(2)
        };
        
        setPredictionData(mockData);
        setMetrics(mockMetrics);
        setIsLoading(false);
      }, 1500);
    } catch (error) {
      console.error('Error making prediction:', error);
      setIsLoading(false);
      setPredictionData(null);
      setMetrics(null);
    }
  };

  return (
    <Container fluid="md">
      <Row>
        <Col md={12}>
          <Card className="mb-4">
            <Card.Header>
              <h3>Bitcoin Price Prediction</h3>
            </Card.Header>
            <Card.Body>
              <Form>
                <Row className="mb-3">
                  <Col md={4}>
                    <Form.Group>
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
                    <Form.Group>
                      <Form.Label>Window Size</Form.Label>
                      <Form.Control 
                        type="number" 
                        value={windowSize} 
                        onChange={(e) => setWindowSize(parseInt(e.target.value) || 0)}
                        min="1"
                        max="100"
                        disabled={selectedModel === 'naive'}
                      />
                    </Form.Group>
                  </Col>
                  <Col md={4}>
                    <Form.Group>
                      <Form.Label>Horizon</Form.Label>
                      <Form.Control 
                        type="number" 
                        value={horizon} 
                        onChange={(e) => setHorizon(parseInt(e.target.value) || 0)}
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
                  className="me-2"
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
                
                <Button 
                  variant="outline-secondary"
                  onClick={() => {
                    setPredictionData(null);
                    setMetrics(null);
                  }}
                >
                  Clear Results
                </Button>
              </Form>
            </Card.Body>
          </Card>
        </Col>
      </Row>

      {metrics && (
        <Row className="mb-4">
          <Col md={12}>
            <Card>
              <Card.Header>
                <h4>Model Performance Metrics</h4>
              </Card.Header>
              <Card.Body>
                <Row>
                  <Col md={3}>
                    <Card bg="light" text="dark" className="mb-2">
                      <Card.Body>
                        <Card.Title>MAE</Card.Title>
                        <Card.Text className="display-6">{metrics.mae}</Card.Text>
                        <Card.Text className="text-muted">Mean Absolute Error</Card.Text>
                      </Card.Body>
                    </Card>
                  </Col>
                  <Col md={3}>
                    <Card bg="light" text="dark" className="mb-2">
                      <Card.Body>
                        <Card.Title>RMSE</Card.Title>
                        <Card.Text className="display-6">{metrics.rmse}</Card.Text>
                        <Card.Text className="text-muted">Root Mean Square Error</Card.Text>
                      </Card.Body>
                    </Card>
                  </Col>
                  <Col md={3}>
                    <Card bg="light" text="dark" className="mb-2">
                      <Card.Body>
                        <Card.Title>MAPE</Card.Title>
                        <Card.Text className="display-6">{metrics.mape}%</Card.Text>
                        <Card.Text className="text-muted">Mean Absolute Percentage Error</Card.Text>
                      </Card.Body>
                    </Card>
                  </Col>
                  <Col md={3}>
                    <Card bg="light" text="dark" className="mb-2">
                      <Card.Body>
                        <Card.Title>MASE</Card.Title>
                        <Card.Text className="display-6">{metrics.mase}</Card.Text>
                        <Card.Text className="text-muted">Mean Absolute Scaled Error</Card.Text>
                      </Card.Body>
                    </Card>
                  </Col>
                </Row>
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}

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
                    height: 500,
                    hovermode: 'x unified'
                  }}
                  config={{ responsive: true }}
                />
              </Card.Body>
            </Card>
          </Col>
        </Row>
      )}
      
      {!predictionData && !isLoading && (
        <Row>
          <Col md={12}>
            <Alert variant="info">
              <h4>How to use this dashboard:</h4>
              <ul>
                <li>Select a model from the dropdown</li>
                <li>Adjust window size and horizon parameters</li>
                <li>Click "Generate Prediction" to run the model</li>
                <li>View the results in the chart and performance metrics</li>
              </ul>
            </Alert>
          </Col>
        </Row>
      )}
    </Container>
  );
};

export default BitcoinPredictionDashboard;