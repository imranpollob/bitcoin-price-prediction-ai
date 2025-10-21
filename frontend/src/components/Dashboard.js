import React from 'react';
import { Container, Row, Col, Card } from 'react-bootstrap';
import BitcoinPredictionDashboard from './BitcoinPredictionDashboard';

const Dashboard = () => {
  return (
    <Container fluid>
      <Row>
        <Col md={12}>
          <Card>
            <Card.Header>
              <h2>Bitcoin Price Prediction Dashboard</h2>
            </Card.Header>
            <Card.Body>
              <BitcoinPredictionDashboard />
            </Card.Body>
          </Card>
        </Col>
      </Row>
    </Container>
  );
};

export default Dashboard;