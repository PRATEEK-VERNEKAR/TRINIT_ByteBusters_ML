import React, { useState, useEffect } from 'react';
import Card from './card';

const Feed = () => {
  const [data, setData] = useState([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [cardsPerPage] = useState(4);

  useEffect(() => {
    fetchData();
  }, [currentPage]); // Include currentPage as a dependency
  ; 

  const fetchData = async () => {
    try {
      const response = await fetch(`http://localhost:8000/top_severe`);
      const jsonData = await response.json();
      console.log(jsonData)
      setData(jsonData);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  };

  const indexOfLastCard = currentPage * cardsPerPage;
  const indexOfFirstCard = indexOfLastCard - cardsPerPage;
  const currentCards = data.slice(indexOfFirstCard, indexOfLastCard);

  const paginate = (pageNumber) => setCurrentPage(pageNumber);

  return (
    <div>
      <div className="feed">
        <h3>Severity: 🔴 High 🟠 Medium 🔵 Low</h3>
        {currentCards.map((item) => (
          <Card key={item.id} data={item} />
        ))}
        <div>
          {data.length > cardsPerPage && (
            <ul className="pagination">
              {Array.from({ length: Math.ceil(data.length / cardsPerPage) }).map((_, index) => (
                <li key={index} className={currentPage === index + 1 ? 'active' : ''}>
                  <button onClick={() => paginate(index + 1)}>{index + 1}</button>
                </li>
              ))}
            </ul>
          )}
        </div>
      </div>
    </div>
  );
};

export default Feed;