import React from 'react';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import BitcoinPredictionDashboard from '../BitcoinPredictionDashboard';

const NBEATSModelView = () => {
  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h3>N-BEATS Model</h3>
            </Card.Header>
            <Card.Body>
              <Alert variant="info">
                Neural Basis Expansion Analysis for Interpretable Time Series Forecasting
              </Alert>
              <BitcoinPredictionDashboard modelType="nbeats" />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default NBEATSModelView;