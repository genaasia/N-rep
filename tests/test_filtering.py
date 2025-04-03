import os
import pytest
from pathlib import Path

import sys
sys.path.append("src")
from text2sql import hello
assert hello.message == "hello, world!"
from text2sql.data.schema_filtering import (
    parse_mac_schema,
    parse_m_schema,
    parse_sql_create,
    parse_basic_format,
    parse_datagrip_format
)

# Constants
TEST_OUTPUTS_DIR = "tests/outputs"
FILTERED_OUTPUTS_DIR = "tests/outputs_filtered"
TEST_DB = "california_schools"

# Create filtered outputs directory if it doesn't exist
os.makedirs(FILTERED_OUTPUTS_DIR, exist_ok=True)

# Test filter dictionary
FILTER_DICT = {
    "schools": "keep_all",
    "satscores": ["cds", "dname", "AvgScrRead", "AvgScrMath", "AvgScrWrite"]
}

def read_schema_file(filename: str) -> str:
    """Read a schema file from the outputs directory"""
    with open(os.path.join(TEST_OUTPUTS_DIR, filename), 'r') as f:
        return f.read()

def write_filtered_file(filename: str, content: str):
    """Write filtered schema to the filtered outputs directory"""
    with open(os.path.join(FILTERED_OUTPUTS_DIR, filename), 'w') as f:
        f.write(content)

@pytest.fixture
def mac_schema():
    return read_schema_file(f"{TEST_DB}_mac_schema.txt")

@pytest.fixture
def mac_schema_basic():
    return read_schema_file(f"{TEST_DB}_mac_schema_basic.txt")

@pytest.fixture
def m_schema():
    return read_schema_file(f"{TEST_DB}_m_schema.txt")

@pytest.fixture
def sql_schema():
    return read_schema_file(f"{TEST_DB}_sql.txt")

@pytest.fixture
def basic_schema():
    return read_schema_file(f"{TEST_DB}_basic.txt")

@pytest.fixture
def basic_types_schema():
    return read_schema_file(f"{TEST_DB}_basic_types.txt")

@pytest.fixture
def basic_relations_schema():
    return read_schema_file(f"{TEST_DB}_basic_relations.txt")

@pytest.fixture
def basic_types_relations_schema():
    return read_schema_file(f"{TEST_DB}_basic_types_relations.txt")

@pytest.fixture
def datagrip_schema():
    return read_schema_file(f"{TEST_DB}_datagrip.txt")

def test_mac_schema_filtering(mac_schema):
    """Test filtering of mac-schema format"""
    filtered = parse_mac_schema(mac_schema, FILTER_DICT)
    write_filtered_file(f"{TEST_DB}_mac_schema_filtered.txt", filtered)
    assert "schools" in filtered
    assert "satscores" in filtered
    assert "frpm" not in filtered
    assert "AvgScrRead" in filtered
    assert "AvgScrMath" in filtered
    assert "AvgScrWrite" in filtered

def test_mac_schema_basic_filtering(mac_schema_basic):
    """Test filtering of mac-schema format with basic filtering"""
    filtered = parse_mac_schema(mac_schema_basic, FILTER_DICT)
    write_filtered_file(f"{TEST_DB}_mac_schema_basic_filtered.txt", filtered)
    
    # Check that schools table is kept with all columns
    assert "# Table: schools" in filtered
    assert "(CDSCode," in filtered
    assert "(NCESDist," in filtered
    
    # Check that satscores table is kept with only specified columns
    assert "# Table: satscores" in filtered
    assert "(cds," in filtered
    assert "(dname," in filtered
    assert "(AvgScrRead," in filtered
    assert "(AvgScrMath," in filtered
    assert "(AvgScrWrite," in filtered
    
    # Check that frpm table is not kept
    assert "# Table: frpm" not in filtered
    
    # Check that other satscores columns are not kept
    assert "(rtype," not in filtered
    assert "(sname," not in filtered
    assert "(cname," not in filtered
    assert "(enroll12," not in filtered
    assert "(NumTstTakr," not in filtered
    assert "(NumGE1500," not in filtered

def test_m_schema_filtering(m_schema):
    """Test filtering of m-schema format"""
    filtered = parse_m_schema(m_schema, FILTER_DICT)
    write_filtered_file(f"{TEST_DB}_m_schema_filtered.txt", filtered)
    assert "schools" in filtered
    assert "satscores" in filtered
    assert "frpm" not in filtered
    assert "AvgScrRead" in filtered
    assert "AvgScrMath" in filtered
    assert "AvgScrWrite" in filtered

def test_sql_create_filtering(sql_schema):
    """Test filtering of SQL CREATE format"""
    filtered = parse_sql_create(sql_schema, FILTER_DICT)
    write_filtered_file(f"{TEST_DB}_sql_filtered.txt", filtered)
    assert "CREATE TABLE schools" in filtered
    assert "CREATE TABLE satscores" in filtered
    assert "CREATE TABLE frpm" not in filtered
    assert "AvgScrRead" in filtered
    assert "AvgScrMath" in filtered
    assert "AvgScrWrite" in filtered

def test_basic_format_filtering(basic_schema):
    """Test filtering of basic format"""
    filtered = parse_basic_format(basic_schema, FILTER_DICT)
    write_filtered_file(f"{TEST_DB}_basic_filtered.txt", filtered)
    assert "table 'schools'" in filtered
    assert "table 'satscores'" in filtered
    assert "table 'frpm'" not in filtered
    assert "AvgScrRead" in filtered
    assert "AvgScrMath" in filtered
    assert "AvgScrWrite" in filtered

def test_basic_types_filtering(basic_types_schema):
    """Test filtering of basic format with types"""
    filtered = parse_basic_format(basic_types_schema, FILTER_DICT, include_types=True)
    write_filtered_file(f"{TEST_DB}_basic_types_filtered.txt", filtered)
    assert "table 'schools'" in filtered
    assert "table 'satscores'" in filtered
    assert "table 'frpm'" not in filtered
    assert "AvgScrRead (INTEGER)" in filtered
    assert "AvgScrMath (INTEGER)" in filtered
    assert "AvgScrWrite (INTEGER)" in filtered

def test_basic_relations_filtering(basic_relations_schema):
    """Test filtering of basic format with relations"""
    filtered = parse_basic_format(basic_relations_schema, FILTER_DICT, include_relations=True)
    write_filtered_file(f"{TEST_DB}_basic_relations_filtered.txt", filtered)
    assert "table 'schools'" in filtered
    assert "table 'satscores'" in filtered
    assert "table 'frpm'" not in filtered
    assert "Relations:" in filtered
    assert "satscores.cds -> schools.CDSCode" in filtered

def test_basic_types_relations_filtering(basic_types_relations_schema):
    """Test filtering of basic format with types and relations"""
    filtered = parse_basic_format(basic_types_relations_schema, FILTER_DICT, include_types=True, include_relations=True)
    write_filtered_file(f"{TEST_DB}_basic_types_relations_filtered.txt", filtered)
    assert "table 'schools'" in filtered
    assert "table 'satscores'" in filtered
    assert "table 'frpm'" not in filtered
    assert "AvgScrRead (INTEGER)" in filtered
    assert "Relations:" in filtered
    assert "satscores.cds -> schools.CDSCode" in filtered

def test_datagrip_format_filtering(datagrip_schema):
    """Test filtering of datagrip format"""
    filtered = parse_datagrip_format(datagrip_schema, FILTER_DICT)
    write_filtered_file(f"{TEST_DB}_datagrip_filtered.txt", filtered)
    assert "schools: table" in filtered
    assert "satscores: table" in filtered
    assert "frpm: table" not in filtered
    assert "+ columns" in filtered
    assert "AvgScrRead: INTEGER" in filtered
    assert "AvgScrMath: INTEGER" in filtered
    assert "AvgScrWrite: INTEGER" in filtered
