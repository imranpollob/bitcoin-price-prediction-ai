import React from 'react';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import BitcoinPredictionDashboard from '../BitcoinPredictionDashboard';

const DenseModelView = () => {
  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h3>Dense Neural Network Model</h3>
            </Card.Header>
            <Card.Body>
              <Alert variant="info">
                Dense model with configurable window size and horizon
              </Alert>
              <BitcoinPredictionDashboard modelType="dense" />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default DenseModelView;