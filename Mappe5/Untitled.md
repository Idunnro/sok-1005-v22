```python
import pandas as pd
g = pd.read_csv("https://www.gapminder.org/answers/how-many-are-rich-and-how-many-are-poor/")#reading data
g
```


    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    /tmp/ipykernel_9707/1406206744.py in <module>
          1 import pandas as pd
    ----> 2 g = pd.read_csv("https://www.gapminder.org/answers/how-many-are-rich-and-how-many-are-poor/")#reading data
          3 g


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/parsers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, dialect, error_bad_lines, warn_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)
        608     kwds.update(kwds_defaults)
        609 
    --> 610     return _read(filepath_or_buffer, kwds)
        611 
        612 


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/parsers.py in _read(filepath_or_buffer, kwds)
        460 
        461     # Create the parser.
    --> 462     parser = TextFileReader(filepath_or_buffer, **kwds)
        463 
        464     if chunksize or iterator:


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/parsers.py in __init__(self, f, engine, **kwds)
        817             self.options["has_index_names"] = kwds["has_index_names"]
        818 
    --> 819         self._engine = self._make_engine(self.engine)
        820 
        821     def close(self):


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/parsers.py in _make_engine(self, engine)
       1048             )
       1049         # error: Too many arguments for "ParserBase"
    -> 1050         return mapping[engine](self.f, **self.options)  # type: ignore[call-arg]
       1051 
       1052     def _failover_to_python(self):


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/parsers.py in __init__(self, src, **kwds)
       1865 
       1866         # open handles
    -> 1867         self._open_handles(src, kwds)
       1868         assert self.handles is not None
       1869         for key in ("storage_options", "encoding", "memory_map", "compression"):


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/parsers.py in _open_handles(self, src, kwds)
       1360         Let the readers open IOHanldes after they are done with their potential raises.
       1361         """
    -> 1362         self.handles = get_handle(
       1363             src,
       1364             "r",


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
        556 
        557     # open URLs
    --> 558     ioargs = _get_filepath_or_buffer(
        559         path_or_buf,
        560         encoding=encoding,


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/common.py in _get_filepath_or_buffer(filepath_or_buffer, encoding, compression, mode, storage_options)
        287                 "storage_options passed with file object or non-fsspec file path"
        288             )
    --> 289         req = urlopen(filepath_or_buffer)
        290         content_encoding = req.headers.get("Content-Encoding", None)
        291         if content_encoding == "gzip":


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/site-packages/pandas/io/common.py in urlopen(*args, **kwargs)
        193     import urllib.request
        194 
    --> 195     return urllib.request.urlopen(*args, **kwargs)
        196 
        197 


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
        212     else:
        213         opener = _opener
    --> 214     return opener.open(url, data, timeout)
        215 
        216 def install_opener(opener):


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/urllib/request.py in open(self, fullurl, data, timeout)
        521         for processor in self.process_response.get(protocol, []):
        522             meth = getattr(processor, meth_name)
    --> 523             response = meth(req, response)
        524 
        525         return response


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/urllib/request.py in http_response(self, request, response)
        630         # request was successfully received, understood, and accepted.
        631         if not (200 <= code < 300):
    --> 632             response = self.parent.error(
        633                 'http', request, response, code, msg, hdrs)
        634 


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/urllib/request.py in error(self, proto, *args)
        559         if http_err:
        560             args = (dict, 'default', 'http_error_default') + orig_args
    --> 561             return self._call_chain(*args)
        562 
        563 # XXX probably also want an abstract factory that knows when it makes


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
        492         for handler in handlers:
        493             func = getattr(handler, meth_name)
    --> 494             result = func(*args)
        495             if result is not None:
        496                 return result


    /usr/local/Miniconda3-py39_4.10.3-Linux-x86_64/lib/python3.9/urllib/request.py in http_error_default(self, req, fp, code, msg, hdrs)
        639 class HTTPDefaultErrorHandler(BaseHandler):
        640     def http_error_default(self, req, fp, code, msg, hdrs):
    --> 641         raise HTTPError(req.full_url, code, msg, hdrs, fp)
        642 
        643 class HTTPRedirectHandler(BaseHandler):


    HTTPError: HTTP Error 403: Forbidden



```python
#len(df)

df = df[0]
#df.dropna(df)
df
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_9707/3777409591.py in <module>
          1 #len(df)
          2 
    ----> 3 df = df[0]
          4 #df.dropna(df)
          5 df


    NameError: name 'df' is not defined



```python
from matplotlib import pyplot as plt

fig,ax=plt.subplots()

#adding axis lables:
ax.set_ylabel('Forventet antall leveår')
ax.set_xlabel('BNP per innbygger')

#plotting the function:
ax.scatter(g['gdp_cap'], g['life_exp'],  label='Observasjoner')
ax.legend(loc='lower right',frameon=False)

```




    <matplotlib.legend.Legend at 0x7f09b6938520>




    
![png](output_2_1.png)
    



```python
import numpy as np
from matplotlib import pyplot as plt

fig,ax=plt.subplots()

#adding axis lables:
ax.set_ylabel('Forventet antall leveår')
ax.set_xlabel('BNP per innbygger')

#plotting the function:
ax.scatter(np.log(g['gdp_cap']), g['life_exp'],  label='Observasjoner')
ax.legend(loc='lower right',frameon=False)
```




    <matplotlib.legend.Legend at 0x7f09b67c9880>




    
![png](output_3_1.png)
    



```python
y=g['life_exp']
pd.DataFrame(y)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>life_exp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>43.828</td>
    </tr>
    <tr>
      <th>1</th>
      <td>76.423</td>
    </tr>
    <tr>
      <th>2</th>
      <td>72.301</td>
    </tr>
    <tr>
      <th>3</th>
      <td>42.731</td>
    </tr>
    <tr>
      <th>4</th>
      <td>75.320</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>137</th>
      <td>74.249</td>
    </tr>
    <tr>
      <th>138</th>
      <td>73.422</td>
    </tr>
    <tr>
      <th>139</th>
      <td>62.698</td>
    </tr>
    <tr>
      <th>140</th>
      <td>42.384</td>
    </tr>
    <tr>
      <th>141</th>
      <td>43.487</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 1 columns</p>
</div>




```python
x=pd.DataFrame(np.log(g['gdp_cap']))
x['intercept']=1
x
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gdp_cap</th>
      <th>intercept</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6.882007</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>8.688964</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>8.736066</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>8.475794</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>9.455588</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>137</th>
      <td>7.800399</td>
      <td>1</td>
    </tr>
    <tr>
      <th>138</th>
      <td>8.014782</td>
      <td>1</td>
    </tr>
    <tr>
      <th>139</th>
      <td>7.732268</td>
      <td>1</td>
    </tr>
    <tr>
      <th>140</th>
      <td>7.147726</td>
      <td>1</td>
    </tr>
    <tr>
      <th>141</th>
      <td>6.152114</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>142 rows × 2 columns</p>
</div>




```python
from statsmodels.regression.linear_model import OLS

res=OLS(y,x).fit()

print(res.summary())
```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    /tmp/ipykernel_9707/413881283.py in <module>
    ----> 1 from statsmodels.regression.linear_model import OLS
          2 
          3 res=OLS(y,x).fit()
          4 
          5 print(res.summary())


    ModuleNotFoundError: No module named 'statsmodels'



```python
res.params
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_9707/112540405.py in <module>
    ----> 1 res.params
    

    NameError: name 'res' is not defined



```python
x=np.linspace(min(np.log(g['gdp_cap'])), max(np.log(g['gdp_cap'])), 100)

regression_line=res.params['intercept']+res.params['gdp_cap']*x

ax.plot(x, regression_line,color='red')
fig
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /tmp/ipykernel_9707/2917901244.py in <module>
          1 x=np.linspace(min(np.log(g['gdp_cap'])), max(np.log(g['gdp_cap'])), 100)
          2 
    ----> 3 regression_line=res.params['intercept']+res.params['gdp_cap']*x
          4 
          5 ax.plot(x, regression_line,color='red')


    NameError: name 'res' is not defined



```python

```
