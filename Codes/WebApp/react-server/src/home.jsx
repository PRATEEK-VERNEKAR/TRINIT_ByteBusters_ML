import React from "react";
import home from './home.jpg'
function Home()
{
    return(
        <div className="home">
        <div className="intro">
            <div className="image">
                <img src={home} alt="img" />
            </div>
            <div className="content">
                <h2>EmpowerSafe: Your Source for Actionable Insights</h2>
                <p>At EmpowerSafe, we understand the importance of data in addressing and preventing sexual harassment. That's why our feed page presents information in a structured manner, organized by severity, to provide concerned authorities and organizations with valuable insights and trends.</p>
    
                <h3>Why Severity Matters:</h3>
                <ul>
                    <li><strong>Prioritizing Response:</strong> By arranging data by severity, we enable authorities and organizations to prioritize their response efforts. Incidents requiring immediate attention are highlighted, ensuring that resources are allocated effectively to address the most critical cases.</li>
                    <li><strong>Identifying Trends:</strong> Patterns and trends often emerge when data is analyzed by severity. By closely monitoring the feed, authorities and organizations can identify recurring issues, hotspots, and emerging threats, allowing them to implement targeted interventions and preventive measures.</li>
                    <li><strong>Enhancing Preparedness:</strong> Understanding the severity of sexual harassment incidents helps authorities and organizations better prepare for potential escalation or crisis situations. By staying informed and proactive, they can take preemptive actions to mitigate risks and protect individuals from harm.</li>
                </ul>
    
                <h3>How We Can Help:</h3>
                <p>At EmpowerSafe, we are committed to providing actionable insights that empower authorities and organizations to take decisive action against sexual harassment. Our user-friendly interface and customizable filters make it easy to navigate the feed and extract relevant information efficiently.</p>
    
                <h3>Join the Movement:</h3>
                <p>Together, we can harness the power of data to create safer environments for everyone. Join EmpowerSafe today and be part of the solution in the fight against sexual harassment. Together, we can make a difference.</p>
            </div>
        </div>
    </div>
    
    )
}

export default Home;