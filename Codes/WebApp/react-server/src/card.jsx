import React from 'react';

function Card({ data }) {
  const { text, location, severity, time, commenting, staring, groping } = data;

  const getShadowClass = (severity) => {
    switch (severity) {
      case 'High':
        return 'shadow-inner-red';
      case 'Medium':
        return 'shadow-inner-yellow';
      case 'Low':
        return 'shadow-inner-blue';
      default:
        return '';
    }
  };

  return (
    <div className={`card ${getShadowClass(severity)}`}>
      <div className="card-header">
        <h2>Incident Details</h2>
      </div>
      <div className="card-body">
        <p>{text}</p>
        <ul>
          <li>Severity: {severity}</li>
          <li>Location: {location}</li>
          <li>Time: {time}</li>
          <li>Commenting: {commenting}</li>
          <li>Ogling/Facial Expressions/Staring: {staring}</li>
          <li>Touching /Groping: {groping}</li>
        </ul>
        <p>
          <a href="https://www.rainn.org/" target="_blank" rel="noopener noreferrer">
            Learn More About Sexual Harassment Helpline
          </a>
        </p>
      </div>
    </div>
  );
}

export default Card;