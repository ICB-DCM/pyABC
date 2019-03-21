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
from sqlalchemy import (Column, Integer, DateTime, String,
                        ForeignKey, Float, LargeBinary)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from .bytes_storage import from_bytes, to_bytes

Base = declarative_base()


class BytesStorage(types.TypeDecorator):
    impl = LargeBinary

    def process_bind_param(self, value, dialect):  # pylint: disable=R0201
        return to_bytes(value)

    def process_result_value(self, value, dialect):  # pylint: disable=R0201
        return from_bytes(value)


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
        return ("<ABCSMC(id={id}, start_time={start_time}, "
                "end_time={end_time})>"
                .format(id=self.id, start_time=self.start_time,
                        end_time=self.end_time))


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
        return ("<Population(id={id}, abc_smc_id={abc_smc_id}, t={t}) "
                "nr_samples={nr_samples} eps={eps} "
                "population_end_time={population_end_time}>"
                .format(id=self.id, abc_smc_id=self.abc_smc_id, t=self.t,
                        nr_samples=self.nr_samples,
                        eps=self.epsilon,
                        population_end_time=self.population_end_time))


class Model(Base):
    __tablename__ = 'models'
    id = Column(Integer, primary_key=True)
    population_id = Column(Integer, ForeignKey('populations.id'))
    m = Column(Integer)
    name = Column(String(200))
    p_model = Column(Float)
    particles = relationship("Particle")

    def __repr__(self):
        return ("<Model id={} population_id={} m ={} name={} p_model={}>"
                .format(self.id, self.population_id, self.m, self.name,
                        self.p_model))


class Particle(Base):
    __tablename__ = 'particles'
    id = Column(Integer, primary_key=True)
    model_id = Column(Integer, ForeignKey('models.id'))
    w = Column(Float)
    parameters = relationship("Parameter")
    samples = relationship("Sample")


class Parameter(Base):
    __tablename__ = 'parameters'
    id = Column(Integer, primary_key=True)
    particle_id = Column(Integer, ForeignKey('particles.id'))
    name = Column(String(200))
    value = Column(Float)

    def __repr__(self):
        return "<{} {}={}>".format(self.__class__.__name__,
                                   self.name, self.value)


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
