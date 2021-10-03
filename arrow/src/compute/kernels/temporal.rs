// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! Defines temporal kernels for time and date related functions.

use chrono::{Datelike, Timelike};

use crate::array::*;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
/// Extracts the hours of a given temporal array as an array of integers
pub fn hour<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
where
    T: ArrowTemporalType + ArrowNumericType,
    i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());
    match array.data_type() {
        &DataType::Time32(_) | &DataType::Time64(_) => {
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    match array.value_as_time(i) {
                        Some(time) => b.append_value(time.hour() as i32)?,
                        None => b.append_null()?,
                    };
                }
            }
        }
        &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, _) => {
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    match array.value_as_datetime(i) {
                        Some(dt) => b.append_value(dt.hour() as i32)?,
                        None => b.append_null()?,
                    }
                }
            }
        }
        dt => {
            return {
                Err(ArrowError::ComputeError(format!(
                    "hour does not support type {:?}",
                    dt
                )))
            }
        }
    }

    Ok(b.finish())
}

/// Extracts the years of a given temporal array as an array of integers
/// If the data is provided in `DataType::Date64` format (millisec/microsecs/nanosecs
/// since 1970), only values above 0 are allowed. Values below 0 will result in `None`.
pub fn year<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
where
    T: ArrowTemporalType + ArrowNumericType,
    i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());
    match array.data_type() {
        &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, _) => {
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    match array.value_as_datetime(i) {
                        Some(dt) => b.append_value(dt.year() as i32)?,
                        None => b.append_null()?,
                    }
                }
            }
        }
        dt => {
            return {
                Err(ArrowError::ComputeError(format!(
                    "year does not support type {:?}",
                    dt
                )))
            }
        }
    }

    Ok(b.finish())
}

/// Extracts the centuries of a given temporal array as an array of integers
/// If the data is provided in `DataType::Date64` format (millisec/microsecs/nanosecs
/// since 1970), only values above 0 are allowed. Values below 0 will result in `None`.
pub fn century<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());
    match array.data_type() {
        &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, _) => {
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    match array.value_as_datetime(i) {
                        Some(dt) => {
                            let year = dt.year();
                            let mut century: i32 = if year == 0 {
                                1
                            } else {
                                (year / 100) as i32
                            };
                            if year - century * 100 != 0 {
                                century += year.signum();
                            }
                            b.append_value(century)?
                        },
                        None => {
                            return Err(ArrowError::ComputeError(format!(
                                "Could not extract century from value {:?}", array.value(i))))
                        },
                    }
                }
            }
        }
        dt => {
            return {
                Err(ArrowError::ComputeError(format!(
                    "century does not support type {:?}",
                    dt
                )))
            }
        }
    }

    Ok(b.finish())
}

/// Extracts the decades of a given temporal array as an array of integers
/// As defined in https://www.postgresql.org/docs/10/functions-datetime.html, decade is
/// "The year field divided by 10".
/// If the data is provided in `DataType::Date64` format (millisec/microsecs/nanosecs
/// since 1970), only values above 0 are allowed. Values below 0 will result in `None`.
pub fn decade<T>(array: &PrimitiveArray<T>) -> Result<Int32Array>
    where
        T: ArrowTemporalType + ArrowNumericType,
        i64: std::convert::From<T::Native>,
{
    let mut b = Int32Builder::new(array.len());

    match array.data_type() {
        &DataType::Date32 | &DataType::Date64 | &DataType::Timestamp(_, _) => {
            for i in 0..array.len() {
                if array.is_null(i) {
                    b.append_null()?;
                } else {
                    match array.value_as_datetime(i) {
                        Some(dt) => {
                            b.append_value((dt.year() / 10) as i32)?
                        },
                        None => b.append_null()?,
                    }
                }
            }
        }
        dt => {
            return {
                Err(ArrowError::ComputeError(format!(
                    "decade does not support type {:?}",
                    dt
                )))
            }
        }
    }

    Ok(b.finish())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_temporal_array_date64_hour() {
        let a: PrimitiveArray<Date64Type> =
            vec![Some(1514764800000), None, Some(1550636625000)].into();

        let b = hour(&a).unwrap();
        assert_eq!(0, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(4, b.value(2));
    }

    #[test]
    fn test_temporal_array_date32_hour() {
        let a: PrimitiveArray<Date32Type> = vec![Some(15147), None, Some(15148)].into();

        let b = hour(&a).unwrap();
        assert_eq!(0, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(0, b.value(2));
    }

    #[test]
    fn test_temporal_array_time32_second_hour() {
        let a: PrimitiveArray<Time32SecondType> = vec![37800, 86339].into();

        let b = hour(&a).unwrap();
        assert_eq!(10, b.value(0));
        assert_eq!(23, b.value(1));
    }

    #[test]
    fn test_temporal_array_time64_micro_hour() {
        let a: PrimitiveArray<Time64MicrosecondType> =
            vec![37800000000, 86339000000].into();

        let b = hour(&a).unwrap();
        assert_eq!(10, b.value(0));
        assert_eq!(23, b.value(1));
    }

    #[test]
    fn test_temporal_array_timestamp_micro_hour() {
        let a: TimestampMicrosecondArray = vec![37800000000, 86339000000].into();

        let b = hour(&a).unwrap();
        assert_eq!(10, b.value(0));
        assert_eq!(23, b.value(1));
    }

    #[test]
    fn test_temporal_array_date64_year() {
        let a: PrimitiveArray<Date64Type> =
            vec![Some(1514764800000), None, Some(1550636625000), Some(0), Some(-1)].into();

        let b = year(&a).unwrap();
        assert_eq!(2018, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(2019, b.value(2));
        assert_eq!(1970, b.value(3));
        assert_eq!(false, b.is_valid(4));
    }

    #[test]
    fn test_temporal_array_date32_year() {
        let a: PrimitiveArray<Date32Type> = vec![Some(15147), None, Some(15448), Some(0), Some(-1)].into();

        let b = year(&a).unwrap();
        assert_eq!(2011, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(2012, b.value(2));
        assert_eq!(1970, b.value(3));
        assert_eq!(1969, b.value(4));
    }

    #[test]
    fn test_temporal_array_timestamp_micro_year() {
        let a: TimestampMicrosecondArray =
            vec![Some(1612025847000000), None, Some(1722015847000000), Some(0), Some(-1)].into();

        let b = year(&a).unwrap();
        assert_eq!(2021, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(2024, b.value(2));
        assert_eq!(1970, b.value(3));
        assert_eq!(false, b.is_valid(4));
    }

    #[test]
    fn test_temporal_array_date64_decade() {
        // Corresponding dates:
        // 1514764800000 = Mon, 01 Jan 2018 00:00:00 GMT
        // 1550636625000 = Wed, 20 Feb 2019 04:23:45 GMT
        // 1617389301000 = Fri, 02 Apr 2021 18:48:21 GMT
        // 1577836800000 = Wed, 01 Jan 2020 00:00:00 GMT
        // 0 = Thursday, January 1, 1970 00:00:00 GMT
        let a: PrimitiveArray<Date64Type> =
            vec![Some(1514764800000), None, Some(1550636625000), Some(1617389301000), Some(1577836800000), Some(0)].into();

        let b = decade(&a).unwrap();
        assert_eq!(201, b.value(0)); // year 2018 corresponds to decade 201
        assert_eq!(false, b.is_valid(1));
        assert_eq!(201, b.value(2)); // year 2019 corresponds to decade 201
        assert_eq!(202, b.value(3)); // year 2021 corresponds to decade 202
        assert_eq!(202, b.value(4)); // year 2020 corresponds to decade 202
        assert_eq!(197, b.value(5)); // year 1970 corresponds to decade 197
    }

    #[test]
    fn test_temporal_array_date32_decade() {
        // Corresponding dates:
        // 15147 = Wednesday, June 22, 2011
        // 15448 = Wednesday, April 18, 2012
        // 14147 = Thursday, September 25, 2008
        // 18719 = Friday, April 2, 2021
        // 0 = Thursday, January 1, 1970
        // -1 = Wednesday, December 31, 1969
        let a: PrimitiveArray<Date32Type> = vec![Some(15147), None, Some(15448), Some(14147), Some(18719), Some(0), Some(-1)].into();

        let b = decade(&a).unwrap();
        assert_eq!(201, b.value(0)); // year 2011 is decade 201
        assert_eq!(false, b.is_valid(1));
        assert_eq!(201, b.value(2)); // year 2012 is decade 201
        assert_eq!(200, b.value(3)); // year 2008 is decade 200
        assert_eq!(202, b.value(4)); // year 2021 is decade 202
        assert_eq!(197, b.value(5)); // year 1970 is decade 197
        assert_eq!(196, b.value(6)); // year 1969 is decade 196
    }

    #[test]
    fn test_temporal_array_timestamp_micro_decade() {
        // Corresponding dates:
        // 1577831800000000 = Tue, 31 Dec 2019 22:36:40 GMT
        // 1722015847000000 = Fri, 26 Jul 2024 17:44:07 GMT
        // 1577836800000000 = Wed, 01 Jan 2020 00:00:00 GMT
        // 0 = Thursday, January 1, 1970
        // -1 = Wednesday, December 31, 1969
        let a: TimestampMicrosecondArray =
            vec![Some(1577831800000000), None, Some(1722015847000000), Some(1577836800000000), Some(0), Some(-1)].into();

        let b = decade(&a).unwrap();
        assert_eq!(201, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(202, b.value(2));
        assert_eq!(202, b.value(3));
        assert_eq!(197, b.value(4));
        assert_eq!(false, b.is_valid(5));
    }

    #[test]
    fn test_temporal_array_date64_century() {
        // Corresponding dates:
        // 946684799000 = Fri, 31 Dec 1999 23:59:59 GMT
        // 946684800000 =  Sat, 01 Jan 2000 00:00:00 GMT
        // 1577836800000 = Wed, 01 Jan 2020 00:00:00 GMT
        // 0 = Thursday, January 1, 1970
        let a: PrimitiveArray<Date64Type> =
            vec![Some(946684799000), None, Some(946684800000), Some(1577836800000), Some(0)].into();

        let b = century(&a).unwrap();
        assert_eq!(20, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(20, b.value(2));
        assert_eq!(21, b.value(3));
        assert_eq!(20, b.value(4));
    }

    #[test]
    fn test_temporal_array_date32_century() {
        // Corresponding dates:
        // 15147 = Wednesday, June 22, 2011
        // 10957 = Saturday, January 1, 2000
        // 0 = Thursday, January 1, 1970
        // -1 = Wednesday, December 31, 1969
        // -719163 = Sunday, December 31, (year) 0000
        // -719563 = Saturday, November 27, -0001
        let a: PrimitiveArray<Date32Type> = vec![Some(15147), None, Some(10957), Some(0), Some(-1), Some(-719163), Some(-719563)].into();

        let b = century(&a).unwrap();
        assert_eq!(21, b.value(0)); // year 2011 is century 21
        assert_eq!(false, b.is_valid(1));
        assert_eq!(20, b.value(2)); // year 2000 is century 20
        assert_eq!(20, b.value(3)); // year 1970 is century 21
        assert_eq!(20, b.value(4)); // year 1969 is century 20
        assert_eq!(1, b.value(5)); // year 0000 is century 1
        assert_eq!(-1, b.value(6)); // year -0001 is century -1
    }

    #[test]
    fn test_temporal_array_timestamp_micro_century() {
        // Corresponding dates:
        // 946684799000000 = Fri, 31 Dec 1999 23:59:59 GMT
        // 946684800000000 =  Sat, 01 Jan 2000 00:00:00 GMT
        // 1577836800000000 = Wed, 01 Jan 2020 00:00:00 GMT
        // 0 = Thursday, January 1, 1970
        let a: TimestampMicrosecondArray =
            vec![Some(946684799000000), None, Some(946684800000000), Some(1577836800000000), Some(0)].into();

        let b = century(&a).unwrap();
        assert_eq!(20, b.value(0));
        assert_eq!(false, b.is_valid(1));
        assert_eq!(20, b.value(2));
        assert_eq!(21, b.value(3));
        assert_eq!(20, b.value(4));
    }
}
