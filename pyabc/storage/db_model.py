"""
Database model
==============

We use SQLAlchemy to associate python classes with database tables,
and instances of the classes with rows in the corresponding tables.
SQLAlchemy includes a system that synchronizes state changes, and a system
for expressing database queries in terms of the user-defined classes and
the relationships defined between them.

For further information see also http://docs.sqlalchemy.org.
"""

import datetime
import sqlalchemy.types as types
from sqlalchemy import (Column, Integer, DateTime, String, VARCHAR,
                        ForeignKey, Float, LargeBinary)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from .bytes_storage import from_bytes, to_bytes
from .version import __db_version__

Base = declarative_base()


class BytesStorage(types.TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):  # pylint: disable=R0201
        return to_bytes(value)

    def process_result_value(self, value, dialect):  # pylint: disable=R0201
        return from_bytes(value)


class Version(Base):
    __tablename__ = 'version'
    version_num = Column(VARCHAR(32), primary_key=True,
                         default=str(__db_version__))


class ABCSMC(Base):
    __tablename__ = 'abc_smc'
    id = Column(Integer, primary_key=True)
    start_time = Column(DateTime)
    end_time = Column(DateTime)
    json_parameters = Column(String(5000))
    distance_function = Column(String(5000))
    epsilon_function = Column(String(5000))
    population_strategy = Column(String(5000))
    git_hash = Column(String(120))
    populations = relationship("Population")

    def __repr__(self):
        return (f"<ABCSMC id={self.id}, start_time={self.start_time}, "
                f"end_time={self.end_time}>")

    def start_info(self):
        return f"<ABCSMC id={self.id}, start_time={self.start_time}>"

    def end_info(self):
        duration = self.end_time - self.start_time
        return (f"<ABCSMC id={self.id}, duration={duration}, "
                f"end_time={self.end_time}>")


class Population(Base):
    __tablename__ = 'populations'
    id = Column(Integer, primary_key=True)
    abc_smc_id = Column(Integer, ForeignKey('abc_smc.id'))
    t = Column(Integer)
    population_end_time = Column(DateTime)
    nr_samples = Column(Integer)
    epsilon = Column(Float)
    models = relationship("Model")

    def __init__(self, *args, **kwargs):
        super(Population, self).__init__(**kwargs)
        self.population_end_time = datetime.datetime.now()

    def __repr__(self):
        return (f"<Population id={self.id}, abc_smc_id={self.abc_smc_id}, "
                f"t={self.t}, nr_samples={self.nr_samples}, "
                f"eps={self.epsilon}, "
                f"population_end_time={self.population_end_time}>")


class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    population_id = Column(Integer, ForeignKey('populations.id'))
    m = Column(Integer)
    name = Column(String(200))
    p_model = Column(Float)
    particles = relationship("Particle")

    def __repr__(self):
        return (f"<Model id={self.id}, population_id={self.population_id}, "
                f"m={self.m}, name={self.name}, p_model={self.p_model}>")


class Particle(Base):
    __tablename__ = 'particles'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    w = Column(Float)
    parameters = relationship("Parameter")
    samples = relationship("Sample")
    proposal_id = Column(Integer, default=0)


class Parameter(Base):
    __tablename__ = 'parameters'
    id = Column(Integer, primary_key=True)
    particle_id = Column(Integer, ForeignKey('particles.id'))
    name = Column(String(200))
    value = Column(Float)

    def __repr__(self):
        return f"<Parameter {self.name}={self.value}>"


class Sample(Base):
    __tablename__ = 'samples'
    id = Column(Integer, primary_key=True)
    particle_id = Column(Integer, ForeignKey('particles.id'))
    distance = Column(Float)
    summary_statistics = relationship("SummaryStatistic")


class SummaryStatistic(Base):
    __tablename__ = 'summary_statistics'
    id = Column(Integer, primary_key=True)
    sample_id = Column(Integer, ForeignKey('samples.id'))
    name = Column(String(200))
    value = Column(BytesStorage)
