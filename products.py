from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
import uuid
from config import SQL_DB_PATH
import logging

db_path = f"sqlite:///{SQL_DB_PATH}"
# Define the SQLAlchemy Base class
Base = declarative_base() 

class Products(Base):
    """
    Products table is simulation of a table in the database
    that stores the products. This is used for the search engine.
    Each product has a UUID and some metadata
    """
    __tablename__ = 'products'
    id = Column(String(36), primary_key=True)
    name = Column(String, nullable=False)
    link = Column(String, nullable=False)
    current_price = Column(Integer, nullable=False)
    original_price = Column(Integer, nullable=False)
    discounted = Column(Boolean, default=False)
    thumbnail = Column(String, default="")
    popularity = Column(Float, default=0.5)
    rating = Column(Float, default=3)

    
    def add(self, session: Session):
        """
        Add the product to the database
        Arguments:
            session: SQLAlchemy session object
        """
        unique_id = str(uuid.uuid4())
        self.id = unique_id
        session.add(self)
        try:
            session.commit()
        except Exception as e:
            session.rollback()
            logging.error("Error adding product to database: ", e)
            raise e


    def __repr__(self):
        return (
            f"<Product(id={self.id}, name='{self.name}', link='{self.link}', "
            f"current_price={self.current_price}, original_price={self.original_price}, "
            f"discounted={self.discounted}, thumbnail='{self.thumbnail}', "
            f"popularity={self.popularity}, rating={self.rating})>"
        )


engine = create_engine(db_path)
Base.metadata.create_all(engine) # type: ignore
Session = sessionmaker(bind=engine)
session = Session()

def test():
    product = Products(
        name = "Test Product",
        link = "https://www.google.com",
        current_price = 100,
        original_price = 200,
    ) # type: ignore
    product.add(session=session)
    print(product)


if __name__ == "__main__":
    test()


