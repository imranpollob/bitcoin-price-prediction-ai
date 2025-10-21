import React from 'react';
import { Container, Row, Col, Card, Alert } from 'react-bootstrap';
import BitcoinPredictionDashboard from '../BitcoinPredictionDashboard';

const EnsembleModelView = () => {
  return (
    <Container>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h3>Ensemble Model</h3>
            </Card.Header>
            <Card.Body>
              <Alert variant="info">
                Combines multiple models with different loss functions
              </Alert>
              <BitcoinPredictionDashboard modelType="ensemble" />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default EnsembleModelView;