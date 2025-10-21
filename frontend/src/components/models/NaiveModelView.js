import React from 'react';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import BitcoinPredictionDashboard from './BitcoinPredictionDashboard';

const NaiveModelView = () => {
  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h3>Naive Forecast Model</h3>
            </Card.Header>
            <Card.Body>
              <Alert variant="info">
                Naive forecast model: ŷₜ = yₜ₋₁ (uses previous timestep value to predict next value)
              </Alert>
              <BitcoinPredictionDashboard modelType="naive" />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default NaiveModelView;