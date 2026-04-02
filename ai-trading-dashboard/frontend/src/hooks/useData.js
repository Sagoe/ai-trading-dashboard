import { useState, useEffect, useCallback, useRef } from "react";
import {
  getHistory, getPrediction, getSentiment,
  getStockPrice, getStockInfo, getPortfolio,
} from "../utils/api";

function useFetch(fetcher, deps = []) {
  const [data, setData]    = useState(null);
  const [loading, setLoad] = useState(true);
  const [error, setError]  = useState(null);
  const mountedRef         = useRef(true);

  const run = useCallback(async () => {
    setLoad(true);
    setError(null);
    try {
      const result = await fetcher();
      if (mountedRef.current) setData(result);
    } catch (e) {
      if (mountedRef.current) setError(String(e));
    } finally {
      if (mountedRef.current) setLoad(false);
    }
  }, deps); // eslint-disable-line

  useEffect(() => {
    mountedRef.current = true;
    setData(null);   // clear on dep change
    run();
    return () => { mountedRef.current = false; };
  }, [run]);

  return { data, loading, error, refetch: run };
}

export const useHistory    = (sym, period)  => useFetch(() => getHistory(sym, period),    [sym, period]);
export const usePrediction = (sym, horizon) => useFetch(() => getPrediction(sym, horizon), [sym, horizon]);
export const useSentiment  = (sym)          => useFetch(() => getSentiment(sym),           [sym]);
export const useStockPrice = (sym)          => useFetch(() => getStockPrice(sym),          [sym]);
export const useStockInfo  = (sym)          => useFetch(() => getStockInfo(sym),           [sym]);
export const usePortfolio  = ()             => useFetch(() => getPortfolio(),              []);
