import React from 'react';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import BitcoinPredictionDashboard from '../BitcoinPredictionDashboard';

const LSTMModelView = () => {
  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h3>LSTM Model</h3>
            </Card.Header>
            <Card.Body>
              <Alert variant="info">
                Long Short-Term Memory recurrent neural network for sequential data
              </Alert>
              <BitcoinPredictionDashboard modelType="lstm" />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default LSTMModelView;