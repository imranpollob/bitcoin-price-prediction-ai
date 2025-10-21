import React from 'react';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import BitcoinPredictionDashboard from '../BitcoinPredictionDashboard';

const Conv1DModelView = () => {
  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h3>Conv1D Model</h3>
            </Card.Header>
            <Card.Body>
              <Alert variant="info">
                1D Convolutional Neural Network for sequence modeling
              </Alert>
              <BitcoinPredictionDashboard modelType="conv1d" />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Conv1DModelView;